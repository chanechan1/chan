API_KEY='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJaZlg3NGJObUNDUDhBZWI2elQ3MldoIiwiaWF0IjoxNjk4NDgxMTEzLCJleHAiOjE3MDAyMzMyMDAsInR5cGUiOiJhcGlfa2V5In0.EDbJYB23JVxxDjdn_TLBWUjq8-sV9iRVP4N8PUG3-9E'

##시간대별로의 모델
goodmodel=[0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, 3, 1, 3, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0]
worstmodel=[0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0]

###최고2개 최악2개
good2model=[[0, 1], [0, 1], [0, 1], [0, 1],
            [0, 1], [0, 1], [0, 1], [1, 0],
            [1, 0], [3, 1], [3, 2], [3, 1],
            [1, 3], [3, 1], [0, 1], [1, 0],
            [3, 2], [1, 2], [0, 1], [0, 1],
            [0, 1], [0, 1], [0, 1], [0, 1]]

worst2model=[[0, 1], [0, 1], [0, 1], [0, 1],
             [0, 1], [0, 1], [4, 2], [4, 2],
             [4, 2], [4, 2], [4, 1], [4, 0],
             [4, 0], [4, 2], [4, 2], [4, 2],
             [4, 0], [4, 0], [0, 1], [0, 1],
             [0, 1], [0, 1], [0, 1], [0, 1]]
