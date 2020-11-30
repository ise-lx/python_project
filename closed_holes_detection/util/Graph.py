

# max_value = 9999
# row0 = [0, 7, max_value, max_value, max_value, 5]
# row1 = [7, 0, 9, max_value, 3, max_value]
# row2 = [max_value, 9, 0, 6, max_value, max_value]
# row3 = [max_value, max_value, 6, 0, 8, 10]
# row4 = [max_value, 3, max_value, 8, 0, 4]
# row5 = [5, max_value, max_value, 10, 4, 0]
# maps = [row0, row1, row2, row3, row4, row5]
# graph = Graph(maps)
# print('邻接矩阵为\n%s' % graph.maps)
# print('节点数据为%d，边数为%d\n' % (graph.nodenum, graph.edgenum))
# print('------最小生成树kruskal算法------')
# print(graph.kruskal())
# print('------最小生成树prim算法')
# print(graph.prim())