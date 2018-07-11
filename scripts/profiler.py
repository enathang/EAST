import cProfile
import importlib
import pstats

name = 'multigpu_train'
print 'importing',name,'to profile'
lib = importlib.import_module(name)
print 'running profiler'
cProfile.run("lib.main()", "restats")

print 'importing stats'
p = pstats.Stats('restats')
p.strip_dirs().sort_stats(-1)

# sort by cum time spent and print top n
p.sort_stats('cumtime').print_stats(50)

