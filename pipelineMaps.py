import numpy as np
from numpy.linalg import norm
import cv2

"""In all the routines below, we assume the rectangle coordinates are
sorted in clockwise order, and they have been preprocessed so that the
"top-left" coordinate is first, as in these examples.


  0 *--------* 1         0 *                 1 *
    |        |            / `                 / `* 2
  3 *--------* 2       3 *   `               /  /
                          `   `             /  /
                           `   * 1       0 *  /
                            ` /             `* 3
                             * 2           
"""
def get_angle ( line_segment ):
    """Angle of the line segment and an array of its cosine and sin
    
    Parameters:
       line_segment: A 1x2 np array of the directed line segment (x,y)
    Returns:
       angle: The scalar angle of the directed line segment (in radians)
       cos_sin: A 1x2 np array containing the cosine and sine of that angle
"""
    angle =  np.arctan2( line_segment[1], line_segment[0] )
    cos_sin = np.array([np.cos(angle), np.sin(angle)])
    return angle, cos_sin

def shrink_01_32( shrunk, rect, reference_lengths, shrink_ratio):
    """ Modify four rectangle points by shrinking the <p0,p1> and <p2,p3>
    segments.

    Parameters:
       shrunk: A 4x2 np array containing the four coordinates of the rectangle
               to be modified
       rect:  A 4x2 np array containing the four coordinates of the original
              rectangle
       reference_lengths: A length 4 list containing the lengths of the 
                          (pre-shrink) four rectangle sides
       shrink_ratio: A scalar in (0, 0.5) indicating by how much to move each 
                     point along the line segment representing a rectangle 
                     side. 
    Returns:
       Nothing. Called for the side-effect on shrunk: The sides <p0,p1> and 
       <p2,p3> are moved "inward" by the relative factor shrink_ratio.
    """
    # Remember that numpy is pass by reference, 
    # so the change to shrunk takes a side-effect in the caller, too

    # <p0,p1>
    cos_sin = get_angle(rect[1] - rect[0])[1]
    shrunk[0] += shrink_ratio * reference_lengths[0] * cos_sin
    shrunk[1] -= shrink_ratio * reference_lengths[1] * cos_sin
    # <p3,p2>
    cos_sin = get_angle(rect[2] - rect[3])[1] 
    shrunk[3] += shrink_ratio * reference_lengths[3] * cos_sin
    shrunk[2] -= shrink_ratio * reference_lengths[2] * cos_sin

def shrink_03_12( shrunk, rect, reference_lengths, shrink_ratio ):
    """ Modify four rectangle points by shrinking the <p0,p3> and <p1,p2>
    segments.

    Parameters:
       shrunk: A 4x2 np array containing the four coordinates of the rectangle
               to be modified
       rect:  A 4x2 np array containing the four coordinates of the original
              rectangle
       reference_lengths: A length 4 list containing the lengths of the 
                          (pre-shrink) four rectangle sides
       shrink_ratio: A scalar in (0, 0.5) indicating by how much to move each 
                     point along the line segment representing a rectangle 
                     side. 
    Returns:
       Nothing. Called for the side-effect on shrunk: The sides <p0,p3> and 
       <p1,p2> are moved "inward" by the relative factor shrink_ratio.
    """
    # Remember that numpy is pass by reference, 
    # so the change to shrunk takes a side-effect in the caller, too

    # <p0,p3>
    cos_sin = get_angle(rect[3]-rect[0])[1]
    shrunk[0] += shrink_ratio * reference_lengths[0] * cos_sin[::-1]
    shrunk[3] -= shrink_ratio * reference_lengths[3] * cos_sin[::-1]
    # <p1,p2>
    cos_sin = get_angle(rect[2]-rect[1])[1]
    shrunk[1] += shrink_ratio * reference_lengths[1] * cos_sin[::-1]
    shrunk[2] -= shrink_ratio * reference_lengths[2] * cos_sin[::-1]


def shrink_rect( rect, shrink_ratio=0.3 ):
    """ Shrink the edges of a rectangle by a fixed relative factor. The
        effect should be equivalent to scaling the height and width of a
        rotated box represented as a center, size, and rotated angle.
    
    Parameters:
        rect: A 4x2 numpy array indicating the coordinates of the four rectangle
              vertices
        shrink_ratio: A scalar in (0,0.5) indicating by how much to move each 
                      point along the line segment representing a rectangle 
                      side. [default 0.3
    Returns:
        shrunk: A 4x2 numpy array with the modified rectangle points
    """

    # Modeled on Eq. (3) in Zhou et al. (EAST), but the mod is outside the +/- 1
    # due to Python's zero-based indexing
    reference_lengths = [ min( norm( rect[c] - rect[(c+1)%4] ),
                               norm( rect[c] - rect[(c-1)%4] ) ) 
                          for c in range(4) ]

    shrunk = rect.copy().astype(np.float32) # Create a clean copy for mutation

    # Find the longer pair of edges --- 
    # {<p0,p1>,<p3,p2>} versus {<p0,p3>,<p1,p2>}
    len_01_32 = norm(rect[0] - rect[1]) + norm(rect[3] - rect[2])
    len_03_12 = norm(rect[0] - rect[3]) + norm(rect[1] - rect[2])

    # Move the longer axes first then shorter axes 
    if len_01_32 > len_03_12:
        # Longer axes
        shrink_01_32( shrunk, rect, reference_lengths, shrink_ratio)
        # Shorter axes
        shrink_03_12( shrunk, rect, reference_lengths, shrink_ratio)
    else:
        # Longer axes
        shrink_03_12( shrunk, rect, reference_lengths, shrink_ratio)
        # Shorter axes
        shrink_01_32( shrunk, rect, reference_lengths, shrink_ratio)

    return shrunk


def dist_to_line(p0, p1, points):
    """ Calculate the distance of points to the line segment <p0,p1> """
    return np.abs( np.cross(p1-p0, points-p0) / norm( p1-p0 ) )


def generate_maps(image_size, rects):
    """ Generate the label maps for training from the preprocessed rectangles 
        intersecting the cropped subimage. 

    Parameters:
       image_size: A two-element tuple [image_height,image_width]
       rects: An 4x2xN numpy array containing the coordinates of the four 
              rectangle vertices. The zeroth dimension runs clockwise around the
              rectangle (as given by sort_rectangle), the first dimension is 
              (x,y), and the last dimension is the particular rectangle.
    Returns:
       score_map: An image_size/4 array of ground truth labels (in {0,1}) for 
                  shrunk versions of the given rectangles
       geo_map: An image_size/4 x 5 array of geometries for the shrunk 
                rectangles; the final dimension contains the distances to the 
                top, right, bottom, and left rectangle eges, as well as the 
                oriented angle in (-pi/2,pi/2)
"""

    # ---------------------------------------------------------------------------
    # Set up return values 
    
    # Where a given rectangle is located
    rect_mask = np.zeros( image_size, dtype=np.uint8) 

    # Pixel-wise positive/negative class indicators for loss calculation
    score_map  = np.zeros( image_size, dtype=np.uint8 ) 
    
    # Distances to four rectangle edges and angle
    geo_map = np.zeros( [image_size[0],image_size[1],5], dtype=np.float32)

    # Which pixels are used or ignored during training
    #training_mask = np.ones( image_size, dtype=uint8 )

    #---------------------------------------------------------------------------
    # Iterate over rectangles:

    for r in xrange(rects.shape[2]):
        rect = rects[:,:,r]
        # Shrink the rectangle, and put in a fillPoly-friendly format
        shrunk_rect = shrink_rect( rect ).astype(np.int32)[np.newaxis,:,:]

        cv2.fillPoly(score_map, shrunk_rect, 1) # Set ground truth pixels to detect
        cv2.fillPoly(rect_mask, shrunk_rect, 1) # Invariant: All 0 before this
        
        # If we wanted to ignore rectangles that were too small, we might do so here    
        #rect_h = min( norm( rect[0]-rect[3]), norm(rect[1]-rect[2]))
        #rect_w = min( norm( rect[0]-rect[1]), norm(rect[2]-poly[3]))
        #if min(rect_h, rect_w) < MIN_POLY_SIZE:
        #    cv2.fillPoly(training_mask, rect.astype(np.int32)[np.newaxis, :, :], 0)

        xy_in_rect = np.argwhere( rect_mask == 1 )
        cols = xy_in_rect[:,0]
        rows = xy_in_rect[:,1]

        # TODO: The original argman/EAST code does not do this, but I think 
        # it's important to ignore the pixels within the ground truth rectangle
        # that were shrunk away, rather than treating them as "negative" labels
                
        # top, right, down, left, angle
        geo_map[rows,cols,0] = dist_to_line( rect[0], rect[1], xy_in_rect) 
        geo_map[rows,cols,1] = dist_to_line( rect[1], rect[2], xy_in_rect)
        geo_map[rows,cols,2] = dist_to_line( rect[2], rect[3], xy_in_rect)
        geo_map[rows,cols,3] = dist_to_line( rect[3], rect[0], xy_in_rect)
        geo_map[rows,cols,4] = get_angle( rect[1] - rect[0])[0]

        rect_mask.fill(0) # Restore invariant

        # If we decide never to modify training_mask,
        # we don't need to generate/return it.
        #
        # That said, perhaps we'd want to track the rectangles labeled
        # "unreadable" and exclude them

    # I'm fairly certain the loss function will want the score to be a
    # float. We store it intermediately to conserve space, and only
    # convert afer downsampling.
    return score_map[::4,::4].astype(np.float32), \
        geo_map[::4,::4,:]
