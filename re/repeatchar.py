import re

doublespace_pattern = re.compile('\s+')
repeatchars_pattern = re.compile('(\w)\\1{2,}')
# 문자가 2개 연속으로 나오는 것 (\\1 ( == \1) : 첫번 째 그룹의 값을 쓰겠다. 즉, \w가 2번 연속으로 나오는 것) 

def repeat_normalize(sent, num_repeats=2):
    if num_repeats > 0:
        sent = repeatchars_pattern.sub('\\1' * num_repeats, sent)
    sent = doublespace_pattern.sub(' ', sent)
    return sent.strip()
    
    
print(repeat_normalize("하하하하하하하하하하헤헤헤헤헤헤헤이이이이이후오어"))
