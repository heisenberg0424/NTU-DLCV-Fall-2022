f = open("val.csv", "r")
f.readline()
cnt=0.0
good=0.0
while(True):
    a = f.readline()
    if(not a):
        break;
    cnt+=1
    truth = a.split('_')[0]
    guess = a.split(',')[-1]
    if(int(truth)==int(guess)):
        good+=1
print('{}/{}, acc:{}'.format(good,cnt,good/cnt))