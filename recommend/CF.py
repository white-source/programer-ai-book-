#协同过滤
def getRatingInfomation(ratings):
    rates = []
    for line in ratings:
        rate = line.split("\t")
        rates.append([int(rate[0]),int(rate[1]),int(rate[2])])
    return rates
#生成用户评分的数据结构
'''
输入：索引数据[[2,1,5],[2,4,2]...]
输出：1 用户打分字段 2 电影字典
使用字典：key-用户id,values-用户打分
rate_dict[2]=[(1,5),(4,2)] 用户2对电影1的打分是5，对电影4的打分是2
'''
def createUserRankDic(rates):
    user_rate_dic = {}
    item_to_user = {}
    for i in rates:
        user_rank = (i[1],i[2])
        if i[0] in user_rate_dic:
            user_rate_dic[i[0]].append(user_rank)
        else:
            user_rate_dic[i[0]] = user_rank
        if i[i] in item_to_user:
            item_to_user[i[1]].append(i[0])
        else:
            item_to_user[i[i]] = [i[0]]
    return user_rate_dic,item_to_user

'''
    格式化成字典数据：
    1 用户字典：dic[用户Id]=[(电影ID，电影评分)...]
    2 电影字典：dic[电影Id]=[用户ID1,用户ID2...]    
'''
def  calcNearestNeighor(userid,test_dic,test_item_to_user):
    current_user_item_data = test_dic[userid]
    users = test_dic.keys()
    for item in current_user_item_data.keys():
        for user in users:
            test_dic[users]


def similarUser(current_user,user2,test_dic):
    current_user_item_list = test_dic[current_user]
    user2_item_list = test_dic[user2]
    for current_tuple_item in current_user:
        print()
    return 0


def similarItem():
    return 0


'''
读取文件列表
'''
def readFile(file_name):
    f=open(file_name,"r",encoding='utf-8')
    line=[]
    line=f.readlines()
    f.close()
    return line

'''
1 基于用户的协同过滤
输入：文件名，用户id,邻居数量
输出：推荐的电影id,输入用户的电影列表，电影对用户的序列表，邻居列表
'''
def recommendByUserCF(file_name,userId,k=5):
    test_content = readFile(file_name)
    #将文件数据格式为二位数组List[[用户ID，电影ID，电影评分]...]
    test_rates = getRatingInfomation(test_content)
    '''
    格式化成字典数据：
    1 用户字典：dic[用户Id]=[(电影ID，电影评分)...]
    2 电影字典：dic[电影Id]=[用户ID1,用户ID2...]    
    '''
    test_dic,test_item_to_user = createUserRankDic(test_rates)
    #寻找K个最相近 用户
    neighbors = calcNearestNeighor(userid,test_dic,test_item_to_user)[:k]
    recommend_dic = {}
    for neighbor in neighbors:
        neighbor_user_id = neighbor[1]
        movies = test_dic[neighbor_user_id]
        for movie  in movies:
            if movie[0] not in recommend_dic:
                recommend_dic[movie[0]] = neighbor[0]
            else:
                recommend_dic[movie[0]]+= neighbor[0]
    #建立推荐列表
    recommend_list =[]
    for key in  recommend_dic:
        recommend_list.append(recommend_dic[key],key)
    recommend_list.sort(reverse=True)
    user_movies = [i[0] for i in test_dic[userId]]
    return [i[1] for i in recommend_list],user_movies,test_item_to_user,neighbors


