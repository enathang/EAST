'''
Calculates the percentage of polygons that have four vertices
'''

import importlib

def main(dir):
    converter = importlib.import_module('convert_data')
    files = converter.loadGroundTruthFile(dir)
    num_rectangles = 0
    num_polygons = 0
    for file in files:
        polygons = converter.loadGroundTruths(file)
        num_polygons += len(polygons)
        for poly in polygons:
            if (len(poly) == 5): # because len(poly)==num_vertices+1
                num_rectangles += 1

    print 'num_rect:', num_rectangles
    print 'num_poly:', num_polygons
    print 'percent:', float(num_rectangles)/num_polygons

if __name__ == "__main__":
  main("../ground_truths/")
        
        
    
