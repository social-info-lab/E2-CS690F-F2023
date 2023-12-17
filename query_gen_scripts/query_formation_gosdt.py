import json

def form_queries(generation_mode='tfidf', regularization=0.05, num_attr=200):     

    # generation_mode = 'tfidf'
    # num_augment = 20000
    gosdt_folder = f'./data/decision-trees/gosdt/gosdt-guesses/'
    gosdt_model_path = f'{gosdt_folder}2020-10-{generation_mode}-{regularization}-{num_attr}-gosdt-tree.txt'
   
    gen_query_path = f'./data/generated-queries/gosdt-guesses/2020-10-{generation_mode}-{regularization}-{num_attr}-gosdt.json'
    with open(gosdt_model_path, "r+") as file1:
        l = file1.readlines()
    querys=list()
    for i in range(12,len(l),5):
        
        queryLine = l[i].strip()
        print(queryLine)
        labelLine = l[i+1].strip()
        words = queryLine.split(" ")
        label = labelLine.split()
        ind = words.index('if')
        queryWords = list()
        word=""
        # for j in range(ind+1,len(words)):
        j=ind+1
        while(j<len(words)):
            if words[j]=="then:":
                break
            #print(words[j])
            word += words[j]
            if(words[j+1] == '='):
                
                if '<' in word:
                    queryWords.append(word[:word.index('<')])
                    word = ""
                    j+=4
                else:
                    word+=" "
                    j+=1
            else:
                if '<' in word:
                    queryWords.append('!'+word[:word.index('<')])
                    word=""
                    j+=4
                else:
                    word+=" "
                    j+=1

        # Writing the queryResults as 0.0 for now. Since they are in the format expected by query_testing.py Script
        queryResults=[[0.0,0.0]]
        if label[2]=='1':
            labels=["Relevant"]
        else:
            labels=["Not Relevant"]
        labels.append(queryResults)
        queryWords.append(labels)
        querys.append(queryWords)
        # jsonFile = json.dumps(querys)
    # print(querys)
    with open(gen_query_path, "w") as outfile:
        json.dump(querys, outfile)

    
    

def main(generation_mode='tfidf', regularization=0.05, num_attr=200):     

    # generation_mode = 'tfidf'
    # num_augment = 20000
    gosdt_folder = f'./data/decision-trees/gosdt/gosdt-guesses/'
    gosdt_model_path = f'{gosdt_folder}2020-10-{generation_mode}-{regularization}-{num_attr}-gosdt-tree.txt'
   
    gen_query_path = f'./data/generated-queries/gosdt-guesses/2020-10-{generation_mode}-{regularization}-{num_attr}-gosdt.json'
    with open(gosdt_model_path, "r+") as file1:
        l = file1.readlines()
    querys=list()
    for i in range(12,len(l),5):
        
        queryLine = l[i].strip()
        print(queryLine)
        labelLine = l[i+1].strip()
        words = queryLine.split(" ")
        label = labelLine.split()
        ind = words.index('if')
        queryWords = list()
        word=""
        # for j in range(ind+1,len(words)):
        j=ind+1
        while(j<len(words)):
            if words[j]=="then:":
                break
            #print(words[j])
            word += words[j]
            if(words[j+1] == '='):
                
                if '<' in word:
                    queryWords.append(word[:word.index('<')])
                    word = ""
                    j+=4
                else:
                    word+=" "
                    j+=1
            else:
                if '<' in word:
                    queryWords.append('!'+word[:word.index('<')])
                    word=""
                    j+=4
                else:
                    word+=" "
                    j+=1

        # Writing the queryResults as 0.0 for now. Since they are in the format expected by query_testing.py Script
        queryResults=[[0.0,0.0]]
        if label[2]=='1':
            labels=["Relevant"]
        else:
            labels=["Not Relevant"]
        labels.append(queryResults)
        queryWords.append(labels)
        querys.append(queryWords)
        # jsonFile = json.dumps(querys)
    # print(querys)
    with open(gen_query_path, "w") as outfile:
        json.dump(querys, outfile)

    
    

# if __name__ == "__main__":
#     main(regularization=0.0001, num_attr=10000)