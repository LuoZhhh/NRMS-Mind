import torch
import argparse
import pandas as pd

from utils import UserDataset
from torch.utils.data import DataLoader
from train import train
from transformers import BertTokenizer
from model import News_Encoder, User_Encoder






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NRMS for MIND Recommendation")

    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--model', type=str, default='bert-mini', choices=['bert-mini', 'bert-medium'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--history_limit', type=int, default=100, help="limit the length of user history")
    parser.add_argument('--num_neg', type=int, default=4, help="number of negative samples")
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--news_path', type=str, default="/home/wuyao/luozihan/NRMS/MIND/MINDsmall_train/news.tsv")
    parser.add_argument('--behavior_path', type=str, default="/home/wuyao/luozihan/NRMS/MIND/MINDsmall_train/behaviors.tsv")

    args = parser.parse_args()
    print(args)
    if args.device > -1:
        device = torch.device("cuda:{}".format(args.device))
    else:
        device = 'cpu'
    
    news = pd.read_csv(args.news_path, delimiter='\s*\t\s*', header=None, index_col=0, engine='python')
    tokenizer_path = args.model
    model_path = args.model
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, padding_side='left')

    model_news = News_Encoder(model_path).to(device)   
    model_user = User_Encoder(model_news).to(device)

    for name, param in model_user.named_parameters():
        if name=="news_encoder.embedding.weight":
            param.requires_grad = False
        else:
            param.requires_grad = True

    print("==== data preprocessing ... ====")
    train_dataset = UserDataset(args.behavior_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    train(news, tokenizer, model_user, model_news, device, train_loader, args)





