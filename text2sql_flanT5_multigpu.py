import os
import json
import torch
import argparse
import torch.optim as optim
import transformers
import torch.nn as nn

from tqdm import tqdm
from tokenizers import AddedToken

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration, MT5ForConditionalGeneration, AutoModel
from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed
from utils.spider_metric.evaluator import EvaluateTool
from utils.load_dataset import Text2SQLDataset
from utils.text2sql_decoding_utils import decode_sqls, decode_natsqls
from mi_estimators import CLUB, CLUBv2, InfoNCE
from utils.mask_text import mask_text, mask_arr

# multi gpu
from accelerate import Accelerator
accelerator = Accelerator()

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")

    parser.add_argument('--mask_ratio', type=float, default=0.40,
                        help='mask_ratio.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size.')
    parser.add_argument('--gradient_descent_step', type=int, default=4,
                        help='perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--device', type=str, default="2",
                        help='the id of used GPU device.')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='learning rate.')
    parser.add_argument('--epochs', type=int, default=128,
                        help='training epochs.')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--save_path', type=str, default="models/text2sql",
                        help='save path of best fine-tuned text2sql model.')
    parser.add_argument('--tensorboard_save_path', type=str, default="tensorboard_log/text2sql",
                        help='save path of tensorboard log.')
    parser.add_argument('--model_name_or_path', type=str, default="t5-3b",
                        help=
                        '''
                        pre-trained model name. 
                        options: 
                            t5-base, https://huggingface.co/t5-base;
                            t5-large, https://huggingface.co/t5-large;
                            t5-3b, https://huggingface.co/t5-3b;
                        ''')
    parser.add_argument('--use_adafactor', action='store_true',
                        help='whether to use adafactor optimizer.')
    parser.add_argument('--mode', type=str, default="train",
                        help='trian, eval or test.')
    parser.add_argument('--train_filepath', type=str, default="data/preprocessed_data/resdsql_train_spider.json",
                        help='file path of test2sql training set.')
    parser.add_argument('--dev_filepath', type=str, default="data/preprocessed_data/resdsql_dev.json",
                        help='file path of test2sql dev set.')
    parser.add_argument('--original_dev_filepath', type=str, default="data/spider/dev.json",
                        help='file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type=str, default="database",
                        help='file path of database.')
    parser.add_argument('--tables_for_natsql', type=str, default="NatSQL/NatSQLv1_6/tables_for_natsql.json",
                        help='file path of tables_for_natsql.json.')
    parser.add_argument('--num_beams', type=int, default=8,
                        help='beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type=int, default=8,
                        help='the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--target_type", type=str, default="sql",
                        help="sql or natsql.")
    parser.add_argument("--output", type=str, default="predicted_sql.txt",
                        help="save file of the predicted sqls.")

    opt = parser.parse_args()

    return opt


def _train(opt):
    set_seed(opt.seed)
    print(opt)

    if opt.tensorboard_save_path is not None:
        writer = SummaryWriter(opt.tensorboard_save_path)
    else:
        writer = None

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    text2sql_tokenizer = T5TokenizerFast.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space=True
    )

    if isinstance(text2sql_tokenizer, T5TokenizerFast):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    train_dataset = Text2SQLDataset(
        dir_=opt.train_filepath,
        mode="train"
    )

    train_dataloder = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
        drop_last=True
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else T5ForConditionalGeneration

    print("initializing text2sql model.")
    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path)
    model.resize_token_embeddings(len(text2sql_tokenizer))
    # initialize mi_estimator
    mi_estimator = InfoNCE(32102, 32102)

    print("finished.")

    # warm up steps (10% training step)
    num_warmup_steps = int(0.1 * opt.epochs * len(train_dataset) / opt.batch_size)
    # total training steps
    num_training_steps = int(opt.epochs * len(train_dataset) / opt.batch_size)
    # save checkpoint for each 1.42857 epochs (about 1.42857*7000=10000 examples for Spider's training set)
    num_checkpoint_steps = int(1.42857 * len(train_dataset) / opt.batch_size)

    # 在此处与主模型任务联合优化
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)] + list(mi_estimator.parameters()),
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if opt.use_adafactor:
        print("Let's use Adafactor!")
        optimizer = Adafactor(
            params=optimizer_grouped_parameters,
            lr=opt.learning_rate,
            scale_parameter=False,
            relative_step=False,
            clip_threshold=1.0,
            warmup_init=False
        )
    else:
        print("Let's use AdamW!")
        optimizer = optim.AdamW(
            params=optimizer_grouped_parameters,
            lr=opt.learning_rate
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # multi gpu
    model, optimizer, train_dataloder, scheduler = accelerator.prepare(model, optimizer, train_dataloder, scheduler)
    mi_estimator = accelerator.prepare(mi_estimator)

    model.train()
    # mi_estimator train
    mi_estimator.train()

    train_step = 0
    for epoch in range(opt.epochs):
        print(f"This is epoch {epoch + 1}.")
        for batch in train_dataloder:
            train_step += 1

            batch_inputs = [data[0] for data in batch]
            batch_sqls = [data[1] for data in batch]
            batch_db_ids = [data[2] for data in batch]  # unused
            batch_tc_original = [data[3] for data in batch]  # unused

            if epoch == 0:
                for batch_id in range(len(batch_inputs)):
                    print(batch_inputs[batch_id])
                    print(batch_sqls[batch_id])
                    print("----------------------")

            tokenized_inputs = text2sql_tokenizer(
                batch_inputs,
                padding="max_length",
                return_tensors="pt",
                max_length=1,
                truncation=True
            )

            with text2sql_tokenizer.as_target_tokenizer():
                tokenized_outputs = text2sql_tokenizer(
                    batch_sqls,
                    padding="max_length",
                    return_tensors='pt',
                    max_length=256,
                    truncation=True
                )

            encoder_input_ids = tokenized_inputs["input_ids"]
            encoder_input_attention_mask = tokenized_inputs["attention_mask"]

            decoder_labels = tokenized_outputs["input_ids"]
            decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
            decoder_attention_mask = tokenized_outputs["attention_mask"]

            if torch.cuda.is_available():
                encoder_input_ids = encoder_input_ids.cuda()
                encoder_input_attention_mask = encoder_input_attention_mask.cuda()
                decoder_labels = decoder_labels.cuda()
                decoder_attention_mask = decoder_attention_mask.cuda()

            model_outputs = model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_input_attention_mask,
                labels=decoder_labels,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=True
            )

            loss = model_outputs["loss"]

            '''
                MIR start
            '''
            # # 1. get masked seq
            # print("mask_arr")
            # batch_inputs_mask = []
            # for batch_input in batch_inputs:
            #     batch_inputs_mask.append(mask_text(batch_input, opt.mask_ratio))
            #     # 对batch_size == 1 进行单独处理
            #     if opt.batch_size == 1:
            #         batch_inputs_mask.append(mask_text(batch_input, opt.mask_ratio))
            # print(batch_inputs_mask)
            #
            # tokenized_inputs_mask = text2sql_tokenizer(
            #     batch_inputs_mask,
            #     padding="max_length",
            #     return_tensors="pt",
            #     max_length=256,
            #     truncation=True
            # )
            # encoder_input_ids_mask = tokenized_inputs_mask["input_ids"]
            # encoder_input_attention_mask_mask = tokenized_inputs_mask["attention_mask"]
            # if torch.cuda.is_available():
            #     encoder_input_ids_mask = encoder_input_ids_mask.cuda()
            #     encoder_input_attention_mask_mask = encoder_input_attention_mask_mask.cuda()
            #
            # # 2. get masked embedding
            # model_outputs_mask = model(
            #     input_ids=encoder_input_ids_mask,
            #     attention_mask=encoder_input_attention_mask_mask,
            #     labels=torch.cat((decoder_labels, decoder_labels), dim=0) if opt.batch_size == 1 else decoder_labels,
            #     decoder_attention_mask=decoder_attention_mask,
            #     return_dict=True
            # )
            #
            # # 3. get noraml embedding
            # mask_embedding = model_outputs_mask["logits"]
            # sentence_embeddings = model_outputs["logits"]
            #
            # # 4. get lower bound value，max MI(masked, noraml)，
            # mask_embedding = mask_embedding[:, 0, :]
            # sentence_embeddings = sentence_embeddings[:, 0, :]
            # sentence_embeddings = torch.cat((sentence_embeddings, sentence_embeddings), dim=0) if opt.batch_size == 1 else sentence_embeddings
            # mi_bound = -mi_estimator(mask_embedding, sentence_embeddings).mean() # 变分下界值
            #
            # # mi_bound.sum().backward(retain_graph=True)
            # # loss.sum().backward()
            # accelerator.backward(loss + mi_bound)
            #
            # print("text2sql损失:")
            # print(loss.item())
            # print("mi损失:")
            # print(mi_bound.item())

            accelerator.backward(loss)
            '''
                MIR end
            '''

            if scheduler is not None:
                scheduler.step()

            if writer is not None:
                # record training loss (tensorboard)
                writer.add_scalar('train loss', loss.item(), train_step)
                # record learning rate (tensorboard)
                writer.add_scalar('train lr', optimizer.state_dict()['param_groups'][0]['lr'], train_step)

            if train_step % opt.gradient_descent_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if train_step % num_checkpoint_steps == 0 and epoch >= 0:
                print(f"At {train_step} training step, save a checkpoint.")
                os.makedirs(opt.save_path, exist_ok=True)
                # 使用accelerator训练模型，需要先解包再保存
                accelerator.wait_for_everyone()
                accelerator.unwrap_model(model).save_pretrained(save_directory=opt.save_path + "/checkpoint-{}".format(train_step),
                    is_main_process = accelerator.is_main_process,
                    state_dict = accelerator.get_state_dict(model),
                    save_func = accelerator.save
                )
                text2sql_tokenizer.save_pretrained(save_directory=opt.save_path + "/checkpoint-{}".format(train_step))

def _test(opt):
    set_seed(opt.seed)
    print(opt)

    import time
    start_time = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    if opt.target_type == "natsql":
        tables = json.load(open(opt.tables_for_natsql, 'r'))
        table_dict = dict()
        for t in tables:
            table_dict[t["db_id"]] = t

    # initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(
        opt.save_path,
        add_prefix_space=True
    )

    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    dev_dataset = Text2SQLDataset(
        dir_=opt.dev_filepath,
        mode=opt.mode
    )

    dev_dataloder = DataLoader(
        dev_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
        drop_last=False
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.save_path else T5ForConditionalGeneration

    # initialize model
    # model = model_class.from_pretrained(opt.save_path)
    model = AutoModel.from_pretrained(opt.save_path)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    predict_sqls = []
    for batch in tqdm(dev_dataloder):
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]
        batch_tc_original = [data[2] for data in batch]

        tokenized_inputs = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )

        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids=encoder_input_ids,
                attention_mask=encoder_input_attention_mask,
                max_length=256,
                decoder_start_token_id=model.config.decoder_start_token_id,
                num_beams=opt.num_beams,
                num_return_sequences=opt.num_return_sequences
            )

            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
            if opt.target_type == "sql":
                predict_sqls += decode_sqls(
                    opt.db_path,
                    model_outputs,
                    batch_db_ids,
                    batch_inputs,
                    tokenizer,
                    batch_tc_original
                )
            elif opt.target_type == "natsql":
                predict_sqls += decode_natsqls(
                    opt.db_path,
                    model_outputs,
                    batch_db_ids,
                    batch_inputs,
                    tokenizer,
                    batch_tc_original,
                    table_dict
                )
            else:
                raise ValueError()

    new_dir = "/".join(opt.output.split("/")[:-1]).strip()
    if new_dir != "":
        os.makedirs(new_dir, exist_ok=True)

    # save results
    with open(opt.output, "w", encoding='utf-8') as f:
        for pred in predict_sqls:
            f.write(pred + "\n")

    end_time = time.time()
    print("Text-to-SQL inference spends {}s.".format(end_time - start_time))

    if opt.mode == "eval":
        # initialize evaluator
        evaluator = EvaluateTool()
        evaluator.register_golds(opt.original_dev_filepath, opt.db_path)
        spider_metric_result = evaluator.evaluate(predict_sqls)
        print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
        print('exec score: {}'.format(spider_metric_result["exec"]))

        return spider_metric_result["exact_match"], spider_metric_result["exec"]


if __name__ == "__main__":
    opt = parse_option()

    if opt.mode in ["train"]:
        _train(opt)
    elif opt.mode in ["eval", "test"]:
        _test(opt)