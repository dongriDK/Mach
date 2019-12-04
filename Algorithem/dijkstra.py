# -*- coding:utf-8 -*-
import sys

def MakeGRAPH(): # 입력받은 경로비용 테이블을 배열화하는 함수
    GRAPH = []
    with open("/Users/mach/Mach/Make/python/dijkstra.txt",'r') as f:
        while True:
            temp = []
            line = f.readline().split()
            for i in range(len(line)):
                if line[i] == "INF":
                    temp.append(INF)
                else:
                    temp.append(int(line[i]))
            if not line: break
            GRAPH.append(temp)
    return GRAPH

def CalcLinkCost(minlist, Node, shortlist, start, j):
    for key, value in minlist.items():
        # 첫 번째 확장에서는 모두 shortlist에 저장
        if Node[key] == False and j == 0 and value != 0:
            shortlist.update({key:minlist.get(key)})
        # 첫번째 확장이 아니고 무한대가 아니고 이미 색칠된 노드는 제외하고
        # 나머지 노드를 경로 비용 계산
        elif minlist.get(key) != INF and Node[key] == False:
            calnum = minlist.get(key) + addnum.get(start)
            if shortlist.get(key) > calnum:
                shortlist.update({key:calnum})

    minvalues = INF #최대값을 기준으로 작은것들로 업데이트하기 위한 변수
    for key, value in shortlist.items(): # shortlist에서 최소값 찾기
        if value and minvalues > value and not Node[key]:
            minvalues = value

    for key, value in shortlist.items(): # 누적 경로비용 계산 후 저장
        if value == minvalues and not Node[key]:
            addnum.update({key:value})
            start = key
            break

    # INF-1로 바꾼 0을 원래대로
    for key, value in shortlist.items():
        if value == INF-1:
            shortlist.update({key:0})

    return shortlist, start

if __name__ == "__main__":
    INF = sys.maxsize
    GRAPH = MakeGRAPH()

    for num in range(len(GRAPH)): # 라우터 개수만큼 동작
        Node = [False] * len(GRAPH) # 색칠 했는지 안했는지 검사하는 변수
        shortlist = {} # 전체 경로 비용 누적 변수
        addnum = {} # 경로계산에서 더해줘야하는 이전 경로비용
        minlist = {} # 현재 노드의 경로비용
        start = GRAPH[num].index(0)  # 기준이 되는 라우터 정해주는 변수

        # 새라우터로 경로 테이블 만들 때마다 shortlist, addnum 초기화
        for i in range(len(GRAPH)):
            shortlist.update({i:0})
            addnum.update({i:0})
        for j in range(len(GRAPH)): # 확장
            Node[start] = True
            for i in range(len(GRAPH)):
                minlist.update({i:0})
            for k in range(len(Node)):
                if GRAPH[start][k] == INF:
                    minlist.update({k:INF})
                # 최소 경로 비용 계산할 때 0은 걸리적거리니 구분하기 쉽게 INF-1로 대치
                elif GRAPH[start][k] == 0:
                    minlist.update({k:INF-1})
                else:
                    minlist.update({k:GRAPH[start][k]})

            shortlist, start = CalcLinkCost(minlist, Node, shortlist, start, j)
        print(str(num)+"라우트의 라우팅 테이블")
        print(shortlist)
