from utils import prompt_gen
import torch
from transformers import BertTokenizer
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm



def test(file, model_user, model_news, tokenizer, device):
	result=[]
	a=pd.read_csv(file, delimiter='\t',header=None)
	for i in tqdm(range(len(a))):
		with torch.no_grad():
			# load data fomr behavior.tsv
			index=a.iloc[i,0]  # Impression ID
			history=a.iloc[i,3]  # history
			if not isinstance(history, str):
				history = "None"
			history = history.split()
			
			candidate = a.iloc[i, 4]  # impressions
			candidate = candidate.split()
			
			scores = []
			prompt_user = []
			prompt_target = []

			# get user embeddings  
			for j in range(len(history)):
				prompt_user.append(prompt_gen(history[j]))
			user_tokens = tokenizer(prompt_user, return_tensors="pt", padding=True)['input_ids'].to(device)
			logits_user = model_user(user_tokens)[0]

			# get news embeddings
			for j in range(len(candidate)):
				prompt_target.append(prompt_gen(candidate[j]))
			target_tokens = tokenizer(prompt_target, return_tensors="pt", padding=True)['input_ids'].to(device)
			logits = model_news(input_ids=target_tokens)[0]

			# computing similarity
			for j in range(len(candidate)):
				score = torch.dot(logits[j], logits_user)
				scores.append(score.item())
			prediction = np.array(scores)
			result_sort = np.argsort(-prediction)
			result_final = np.empty([len(result_sort)], dtype = int) 
			for j in range(len(result_sort)):
				result_final[result_sort[j]] = j + 1  # start from 1
			text = str(index)+" ["+','.join(str(i) for i in result_final)+"]\n"
			result.append(text)
	
	filename = "prediction.txt"
	with open(filename, 'w') as f:
		f.writelines(result)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--news_path', type=str, default="/home/wuyao/luozihan/NRMS/MIND/MINDsmall_train/news.tsv")
    parser.add_argument('--behavior_path', type=str, default="/home/wuyao/luozihan/Mind/data/test/behaviors.tsv")
    args = parser.parse_args()

    print(args)
    if args.device > -1:
        device = torch.device("cuda:{}".format(args.device))
    else:
        device = 'cpu'
    
    tokenizer_path =  f"bert-mini"
    model_path_bert = f"bert-mini"

    model_user = torch.load('user.pth').to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, padding_side='left')                  
    model_news = torch.load('news.pth').to(device)
    test(args.behavior_path, model_user, model_news, tokenizer, device)
