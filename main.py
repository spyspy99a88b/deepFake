import cv2
import dlib
import numpy as np
import copy

# mouth index to keep emotion
mouth_index = [[60],[61],[62],[63],[64],[65],[66],[67]]
mouth_index_set = set(i[0] for i in mouth_index)

def get_delaunay_triangles_index(points, indices):
    # only construct triangles between hull and mouth
    hull = cv2.convexHull(np.array(points))
    rect = cv2.boundingRect(hull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    points = np.array(points, np.int32)

    delaunay_triangles_index = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        tri_idx = []
        for i, p in enumerate(points):
            if ((pt1[0] == p[0] and pt1[1] == p[1]) 
            or (pt2[0] == p[0] and pt2[1] == p[1])
            or (pt3[0] == p[0] and pt3[1] == p[1])):
                tri_idx.append(indices[i][0])
            if len(tri_idx) == 3:
                delaunay_triangles_index.append(tri_idx)
                break
    return delaunay_triangles_index


def get_triangles(landmarks_points, tri_index):
    pt1 = landmarks_points[tri_index[0]]
    pt2 = landmarks_points[tri_index[1]]
    pt3 = landmarks_points[tri_index[2]]
    return np.array([pt1, pt2, pt3], np.int32)


def warp_triangle(img1, img2, bb1, bb2, t1, t2):
    # https://www.learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/

    img1_cropped = img1[bb1[1]: bb1[1] + bb1[3], bb1[0]: bb1[0] + bb1[2]]

    t1_offset = [
        ((t1[0][0] - bb1[0]), (t1[0][1] - bb1[1])),
        ((t1[1][0] - bb1[0]), (t1[1][1] - bb1[1])),
        ((t1[2][0] - bb1[0]), (t1[2][1] - bb1[1])),
    ]
    t2_offset = [
        ((t2[0][0] - bb2[0]), (t2[0][1] - bb2[1])),
        ((t2[1][0] - bb2[0]), (t2[1][1] - bb2[1])),
        ((t2[2][0] - bb2[0]), (t2[2][1] - bb2[1])),
    ]
    mask = np.zeros((bb2[3], bb2[2], 3), dtype=np.float32)

    cv2.fillConvexPoly(mask, np.int32(t2_offset), (1.0, 1.0, 1.0), cv2.LINE_AA) #16, 0, cv2.LINE_AA

    size = (bb2[2], bb2[3])

    mat = cv2.getAffineTransform(np.float32(t1_offset), np.float32(t2_offset))

    img2_cropped = cv2.warpAffine(
        img1_cropped,
        mat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    img2_cropped = img2_cropped * mask

    # bb2_y = max(bb2[1], 0)

    img2_cropped_slice = np.index_exp[
                         bb2[1]: bb2[1] + bb2[3], bb2[0]: bb2[0] + bb2[2]
                         ]
    img2[img2_cropped_slice] = img2[img2_cropped_slice] * ((1.0, 1.0, 1.0) - mask)
    img2[img2_cropped_slice] = img2[img2_cropped_slice] + img2_cropped

def capture_best_img_from_source(source_video_loc):
    print('Source image preprocessing start.')
    print('Start capturing the largest face image from the source video.')
    
    cap_s = cv2.VideoCapture(source_video_loc)
    length = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
    max_area = 0
    best_source_img = None
    trial = 0

    while True:
        if trial >= length:
            break
        trial += 1
        print('trial:', trial, '/', length)

        success, img = cap_s.read()
        if not success:
            continue
        detects = detector(img)
        if len(detects) != 0:
            det = max(detects, key=lambda x: x.area())
            det_area = det.area()
            if det_area > max_area:
                max_area = det_area
                print('max image area now:', max_area, 'pixels.')
                best_source_img = img
                break

    img_source = copy.deepcopy(best_source_img)

    tri_indices = None
    landmarks_points_source = None
    detects_source = detector(img_source)
    if len(detects_source) != 0:
        det = max(detects_source, key=lambda x: x.area())
        landmarks_source = predictor(img_source, det)

        landmarks_points_source = []
        for point in landmarks_source.parts():
            landmarks_points_source.append((point.x, point.y))
        
        # hull for mouth to keep emotion
        hull_index_ori = cv2.convexHull(np.array(landmarks_points_source), returnPoints=False)
        hull_index = np.concatenate((hull_index_ori, mouth_index))
        landmark_idx_to_list_idx = {e[0]: i for i, e in enumerate(hull_index)}
        points = [landmarks_points_source[i[0]] for i in hull_index]
        tri_indices = get_delaunay_triangles_index(points, hull_index)

    tri_source_lst = []
    bb1_lst = []
    for tri_index in tri_indices:
        tri_source = get_triangles(landmarks_points_source, tri_index)
        tri_source_lst.append(tri_source)
        bb1 = cv2.boundingRect(np.float32([tri_source]))
        bb1_lst.append(bb1)
    
    detects = detector(best_source_img)
    det = max(detects, key=lambda x: x.area())
    # show face boundaries
    cv2.rectangle(best_source_img, (det.left(), det.top()), (det.right(), det.bottom()), (0, 0, 255), 3)

    print('Source image preprocessing done.')
    print('Max image face area:', max_area, 'pixels.')
    print('-------------------------------------')
    print('You wanna have a look? Type y or n.')
    while True:
        option_input = input('')
        # check options
        if (option_input.upper() == 'Y'):
            cv2. imshow('best_source_img', best_source_img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            while True:
                key = cv2.waitKey(0)
                if key in [27, ord('q'), ord('Q')]:
                    cv2.destroyAllWindows()
                    break
            break
        elif (option_input.upper() == 'N'):
            break
        else:
            print('Invalid option, please type y or n.')

    cap_s.release()
    # cv2.destroyAllWindows()
    return landmarks_points_source, tri_indices, img_source, tri_source_lst, bb1_lst,\
        hull_index_ori, hull_index, landmark_idx_to_list_idx

def capture_source_img_from_img(image_path):
    img_source = cv2.imread(image_path)
    tri_indices = None
    landmarks_points_source = None
    # if not success:
    #     print("reading second image error")
    detects_source = detector(img_source)
    if len(detects_source) != 0:
        det = max(detects_source, key=lambda x: x.area())
        landmarks_source = predictor(img_source, det)
        landmarks_points_source = []
        for point in landmarks_source.parts():
            landmarks_points_source.append((point.x, point.y))
        
        hull_index_ori = cv2.convexHull(np.array(landmarks_points_source), returnPoints=False)
        hull_index = np.concatenate((hull_index_ori, mouth_index))
        landmark_idx_to_list_idx = {e[0]: i for i, e in enumerate(hull_index)}
        points = [landmarks_points_source[i[0]] for i in hull_index]
        tri_indices = get_delaunay_triangles_index(points, hull_index)

    tri_source_lst = []
    bb1_lst = []
    for tri_index in tri_indices:
        tri_source = get_triangles(landmarks_points_source, tri_index)
        tri_source_lst.append(tri_source)
        bb1 = cv2.boundingRect(np.float32([tri_source]))
        bb1_lst.append(bb1)
    
    return landmarks_points_source, tri_indices, img_source, \
        tri_source_lst, bb1_lst, hull_index_ori, hull_index, landmark_idx_to_list_idx



if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # source_video_loc = 'test video/dance1.mp4'
    # landmarks_points_source, tri_indices, img_source, tri_source_lst,\
    #  bb1_lst, hull_index_ori, hull_index, landmark_idx_to_list_idx = capture_best_img_from_source(source_video_loc)

    source_image_loc = 'videoAndPics/4.jpg'
    landmarks_points_source, tri_indices, img_source, tri_source_lst,\
     bb1_lst, hull_index_ori, hull_index, landmark_idx_to_list_idx = capture_source_img_from_img(source_image_loc)
    
    video_loc = 'videoAndPics/3.mp4'
    cap = cv2.VideoCapture(video_loc)
    print('Start doing face swapping.')

    frame_init = True
    while True:

        success, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not success:
            continue

        detects = detector(img)
        if len(detects) != 0:
            det = max(detects, key=lambda x: x.area())
            landmarks = predictor(img, det)

            # show face boundaries
            # cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (0, 0, 255), 1)
            # cv2.imshow("face", img)
            # cv2.waitKey(0)
            # show face landmarks
            # for point in landmarks.parts():
            #     cv2.circle(img, (point.x, point.y), 1, (0, 0, 255), 1)

            # show convex hulls
            landmarks_points_target = []
            for point in landmarks.parts():
                landmarks_points_target.append((point.x, point.y))

            hull_target = [landmarks_points_target[i[0]] for i in hull_index]
            original_hull_target = [landmarks_points_target[i[0]] for i in hull_index_ori]
            # frame_init

            if frame_init:
                hull_target_last_frame = np.array(hull_target, np.float32)
                img_gray_previous = copy.deepcopy(img_gray)
                first_frame = True

            hull2_next, *_ = cv2.calcOpticalFlowPyrLK(
                img_gray_previous,
                img_gray,
                hull_target_last_frame,
                np.array(hull_target, np.float32),
                winSize=(101, 101),
                maxLevel=5,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001),
            )

            current_factor = 0.5
            for i, _ in enumerate(hull_target):
                hull_target[i] = current_factor * np.array(hull_target[i]) + (1 - current_factor) * hull2_next[i]

            hull_target_last_frame = np.array(hull_target, np.float32)
            img_gray_previous = img_gray

            img_source_warped = np.copy(img)
            img_source_warped = np.float32(img_source_warped)




            # hulls = cv2.convexHull(np.array(landmarks_points_target))

            # img = cv2.fillPoly(img, [hulls], (255, 0, 0))
            # delaunary triangles
            # get_delaunay_triangles_index(points, img)
            # img_source_warped = np.zeros_like(img)

            # img_source_warped = np.float32(img)
            
            break_check = False
            index = 0
            for tri_index in tri_indices:
                # remove mouth triangles
                if (tri_index[0] in mouth_index_set and tri_index[1] in mouth_index_set and tri_index[2] in mouth_index_set):
                    index += 1
                    continue 
                tri_target = get_triangles(landmarks_points_target, tri_index)
                
                bb2 = cv2.boundingRect(np.float32([tri_target]))
                if bb2[1] < 0:
                    break_check = True
                    break
                warp_triangle(img_source, img_source_warped, \
                    bb1_lst[index], bb2, tri_source_lst[index], tri_target)
                index += 1

            if break_check:
                continue

            mask = np.zeros_like(img_gray, dtype=img.dtype)
            cv2.fillConvexPoly(mask, np.int32(original_hull_target), 255)
            bb = cv2.boundingRect(np.float32(original_hull_target))
            center = (bb[0] + int(bb[2] / 2), bb[1] + int(bb[3] / 2))
            img = cv2.seamlessClone(
                np.uint8(img_source_warped), img, mask, center, cv2.NORMAL_CLONE
            )

        cv2.imshow("face", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
