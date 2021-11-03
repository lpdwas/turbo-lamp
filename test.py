#encoding="utf-8"
from datetime import datetime
from tkinter.constants import N
import jieba,os,re
import pandas as pd 
from gensim.models import Phrases
import numpy as np
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
import pickle
import csv
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

# 语料库处理（切片+bi_gram模型训练）
def slicing (stopword_path): 
    filenames = os.listdir('test/')
    source_data = {}
    din_word = []
    training_data = []
    with open (stopword_path,'r',encoding='utf-8-sig') as f:
        stop = f.read()
        stopword = stop.split()
    for filename in filenames:
        file_size = os.path.getsize('test/'+filename)
        if float(file_size/1000) > 100:
            continue
        pattern = re.compile('\d{4}\d{2}\d{2}')
        date = pattern.findall(filename)[0]
        date = datetime.datetime.strptime(date,'%Y%m%d')
        date = date.strftime('%Y/%#m/%#d')
        print('正在处理'+str(date)+'日的语料', end='\r')
        with open('test/'+filename, 'r', encoding='utf-8-sig') as f:
            data = f.read().replace('\n','')
        words = []
        items = jieba.lcut(data)       
        for n in range(len(items)-1,-1,-1):  
            word= items[n]
            if word in stopword:
                items.pop(n)
                continue
            if '.' in word:
                items.pop(n)
                continue
            if '\uf06c' in word:
                items.pop(n)
                continue
            for ky in word :

                if ky in '0 1 2 3 4 5 6 7 8 9 （ ） 《 》 / 【 】 ：• '.split():
                    items.pop(n)
                    break
        for k in items:
            words.append(k)
            if k in '。 ， ？ ！'.split():
                words.remove(k)
                din_word.append(words)
                training_data.append(words)
                words =[]
        try :
            source_data[date]
        except:
            source_data[date] = []
        source_data[date].append(din_word)
        din_word = []
    return source_data,training_data

# bi-gram模型训练语句
def bi_gram(training_data,model_path):
    phrases_model = Phrases(training_data, min_count=20, threshold=30)  
    phrases_model.save(model_path)

# 进行bi_gram处理
def training (din,model_path): 
    new_din ={}
    a = []
    bigram = Phrases.load(model_path)  # 导入训练好的bi-gram模型
    print('正在训练')
    for key in din.keys():
        print('正在训练{}的语料'.format(key),end='\r')
        new_din[key] = []
        for news_amo in din[key]:
            for real_news in range(len(news_amo)):
                catch = bigram[news_amo[real_news]]  # 将字典中的词导入模型中进行处理
                for i in range(len(catch)):
                    if '_' in catch[i]:
                        catch[i] = catch[i].replace('_','')
                        if not catch[i] in a:
                            a.append(catch[i])
                        jieba.add_word(catch[i])
                new_din[key].append(catch) 
    with open('words.txt','w',encoding='utf-8-sig') as f:
        for n in a:
            f.write(n+'\n')  
    print('训练完成')
    return new_din

# 对字典中的文本进行拼接
def splicing(data):
    sentence =[]
    print('正在拼接', end='\r')
    for words in data:
        sentence.append(' '.join(str(i)for i in words))
    return sentence 

def train(train_list,model_name,mindf):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90 , min_df=0.0022)  # Tfidf模型训练语句 重点改进
    tfidf_X = tfidf_vectorizer.fit(train_list)
    word = tfidf_vectorizer.get_feature_names()
    with open(model_name,'wb') as f: 
        pickle.dump(tfidf_X,f) 
    return word

def Onehot_encode(train_din,model_name, state ,mindf):  
    train_list = []
    for key in train_din.keys():
        train_din[key] = splicing(train_din[key])
        train_list.extend(train_din[key])
        train_din[key] = [' '.join(str(i)for i in train_din[key])]
    print('拼接完成')
    word = train(train_list,model_name,0.008)
    with open(model_name,'rb') as k:
        tfidf_X = pickle.load(k)
    for new_key in train_din.keys():
        train_din[new_key] = tfidf_X.transform(train_din[new_key]).toarray()
        # print(train_din[new_key])
        for i in range(len(train_din[new_key])):
            for j in range(len(train_din[new_key][i])):
                if train_din[new_key][i,j] > mindf:
                    train_din[new_key][i,j] = 1
                else:
                    train_din[new_key][i,j] = 0           
    return word

# 读取市场价格，判断涨跌
def market_data(file_path, proportion):
    data2 = pd.read_csv(file_path, encoding='utf-8')
    market = {}
    for p in range(len(data2)):
        judge =(data2['close'][p] - data2['open'][p])/data2['open'][p]
        if judge >= proportion:  # 当价差比例>输入比例proportion时,判断为上涨
            state = 1  # 当市场上涨时为1
        elif judge <= -proportion:  # 当价差比例<-proportion时，判断为下跌
            state = 0  # 当市场下跌时为0
        else:
            continue
        date = data2['day'][p]
        market[date] = state
    return market  # 涨跌情况存放到以日期为键的字典当中

# 利用datetime库进行日期的换算，从而获得n天后的日期
def date_change(str_date, n_day):
    time_date = datetime.datetime.strptime(str_date,'%Y/%m/%d')
    delta = datetime.timedelta(days=n_day)
    n_days = time_date + delta
    future_days = n_days.strftime('%Y/%#m/%#d')
    return future_days

# 进行Naive Bayes模型训练
def Naive (X_din,Y_din,Naive_model_path,n_day):
    clf = BernoulliNB()
    X_list = []
    Y_list = []
    offset = int(len(Y_din.keys())*0.7)
    for key in list(Y_din.keys())[:offset]: 
        new_key = date_change(key, n_day)
        if key in X_din and new_key in Y_din:  # 通过extend函数将词汇和涨跌表现分别整合
            X_list.extend(X_din[key])
            Y_list.extend([Y_din[new_key]])
    Y = np.array(Y_list)
    X = np.array(X_list)
    clf.fit(X,Y)
    right = clf.score(X,Y)  # 输出拟合的准确度
    with open(Naive_model_path,'wb') as f:  # 对训练出来的模型进行储存
        pickle.dump(clf,f) 
    return offset,right  

# 求日平均词汇量
def analy (din):
    n=0
    for key in din.keys():
        for i in din[key]:
            n = n + len(i)
    words_average = n / len(din.keys())
    return words_average

# 导入训练得到的模型进行预测
def predict_data(new_din, Y_din, Naive_model_path, n_day, offset): 
    y_pre = []
    y_label = []
    predict = []
    all_amo = 0
    n = 0
    with open (Naive_model_path,'rb') as f:
        clf = pickle.load(f)
    for key in list(Y_din.keys())[offset:]:
        new_key = date_change(key, n_day)
        if key in new_din and new_key in Y_din: 
            all_amo += 1
            X = np.array(new_din[key])
            a = clf.predict(X)
            if a == Y_din[key]:
                n += 1
                try:
                    predict = np.array(predict)
                    predict = predict + X 
                except:
                    predict = X
            pro = clf.predict_proba(X)
            y_pre.append(pro[0][1])
            y_label.append(Y_din[new_key])
    try:
        right_catch = n/all_amo
    except:
        right_catch = 0
    
    return y_pre,y_label,predict,right_catch

# 统计特征词出现次数
def count(word_list,encode_din):
    din = {}
    n = 0
    for word in word_list:
        din[word] = 0
        for key in encode_din.keys():
            if encode_din[key][0][n] == 1:
                din[word] = din[word] + 1
        n = n + 1
    return din

# 存放特征词频分布
def word_distribution(filename,new_din):
    with open('data/word_list.pickle','rb') as f:
        word = pickle.load(f)
    g = open ('distribution/'+filename+'.csv', 'a', encoding='utf-8-sig', newline='')
    csv_writer = csv.writer(g)
    for i in range(len(word)):
        word_count = 0
        for key in new_din.keys():
            if new_din[key][0][i] == 1:
                word_count = word_count + 1
        csv_writer.writerow([word[i],word_count])
    print('Done')

# 画出模型的ROC图
def draw(market,y_pre,n_day,right_catch,state):
    if right_catch:
        fpr, tpr, thersholds = roc_curve(market, y_pre, pos_label=1)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = 0
    if state:
        plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')  
        plt.title(str(n_day)+'day ROC')
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc

def __main__(test_dataX, test_dataY, bi_gram_model_path, model_name, Naive_model_path, state, n_day, mindf, proportion):
    # din,training_data = slicing('stopwords.txt')
    # with open ('traindata.pickle','wb') as f:
    #     pickle.dump(din,f)
    # with open ('train_data.pickle','wb') as g:
    #     pickle.dump(training_data,g)
    with open ('traindata.pickle','rb') as f:
        din = pickle.load(f)
    with open ('train_data.pickle','rb') as g:
        training_data = pickle.load(g)
    market = market_data(test_dataY, proportion)
    bi_gram(training_data, bi_gram_model_path)
    # new_din = training(din, bi_gram_model_path)
    # with open ('data/new_din.pickle','wb') as o:
    #     pickle.dump(new_din,o)
    with open ('data/new_din.pickle','rb') as o:
        new_din = pickle.load(o)
    word = Onehot_encode(new_din, model_name, state, mindf)
    if state:
        with open ('data/word_list.pickle','wb') as k:
            pickle.dump(word,k)
    offset,right = Naive(new_din, market, Naive_model_path, n_day)
    word_din = count(word, new_din)
    y_pre,market_Y,predict,right_catch = predict_data(new_din, market, Naive_model_path, n_day, offset)
    roc = draw(market_Y, y_pre, n_day, right_catch,state)
    # with open('data/word_list.pickle','rb') as u:
    #     word_list = pickle.load(u)
    # file = open('count.csv','w',encoding='utf-8-sig',newline='')
    # file_csv = csv.writer(file)
    # file_csv.writerow(['词汇','频率'])      #以csv文件形式导出特征词出现次数
    # for x in range(len(word_list)):
    #     file_csv.writerow([word_list[x],predict[0][x]])
    # print('DONE')
    # print(len(market.keys()))
    # print(word_din)
    return right_catch,roc

""" 1.利用silcing函数对语料库进行切词，再把切好的词按日期存放到字典当中
    2.将切好的词汇放进已训练好的bi-gram模型中进行bi-gram处理，得到新的bi-gram字典
    3.将新的字典进行One-hot encode处理，找到其特征词并转换成One-hot encode
    4.将已有的数据放入Naive Bayes函数内进行模型训练，得到模型并保存
    5.获取需要预测的数据文件，输入predict_data函数中，计算正确率
"""
nday = 2  #需要预测的天数
name = '焦煤'  #预测的品种，目前为焦炭、动力煤、焦煤
best_score = 0
best_roc = 0
for test in [0.10,0.15,0.2,0.25,0.3]:  #得分阈值的选取范围
    for proportion in [0.0010,0.0015,0.0020,0.0025]:  #价格变化比例的选取范围
        right,roc = __main__(test_dataX='data/'+'2021'+'_news.csv', test_dataY=name + '.csv', bi_gram_model_path=name+'.model',
            model_name='save/'+name+'_One_hot_'+str(nday)+'.pickle', Naive_model_path='save/Naivemodel_'+name+'_'+ str(nday) +'.pickle', 
            state=0, n_day=nday, mindf = round(test,2), proportion=round(proportion,3))
        score = right
        if score >= best_score and roc >= best_roc:
                best_roc = roc 
                best_score = score
                best_parameters = {'得分阈值':test , '价格比例':proportion}
print('最好预测正确率为：',best_score,end='\n')
print('该准确率下参数为:',best_parameters,end='\n')
print('ROC指数为:',best_roc)
__main__(test_dataX='data/'+'2021'+'_news.csv', test_dataY=name + '.csv', bi_gram_model_path=name+'.model',
        model_name='save/'+name+'_One_hot_'+str(nday)+'.pickle', Naive_model_path='save/Naivemodel_'+name+'_'+ str(nday) +'.pickle', 
        state=1, n_day=nday, mindf = round(best_parameters['得分阈值'],2), proportion=round(best_parameters['价格比例'],3))