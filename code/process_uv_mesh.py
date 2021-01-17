import numpy as np
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
# Position from Barycentric coords
def getPos(face, b_coords):
    P = b_coords[0] * face[0] + b_coords[1] * face[1] + (1 - b_coords[0] - b_coords[1]) * face[2]
    return P

# Barycentric coords from 
def getBary(face, pt):
    a_P = pt - face[0]
    ab = face[1] - face[0]
    ac = face[2] - face[0]
    total_area = np.linalg.norm(np.cross(ac, ab))/2.0
    v = np.linalg.norm(np.cross(ac, a_P))/2.0/total_area
    w = np.linalg.norm(np.cross(ab, a_P))/2.0/total_area
    u = 1.0 - v - w
    uv_coords = np.array([u, v])
    return uv_coords

def make_bary_texels(side_len):
    side = side_len
    # Parse .obj for UV texture info/geometry info
    f = open("D:\Brown\Senior\CSCI_1470\FINAL\smpl_UV\SMPL_UV_NEUTR.obj", "r")

    uv_pts = []
    pos_pts = []

    f_uv = []
    f_pos = []

    for line in f:
        if line[:1] == "v" and not(line[:2] == "vt"):
            split = line.split()
            pos_pts.append([float(split[1]), float(split[2]), float(split[3])])
        if line[:2] == "vt":
            split = line.split()
            uv_pts.append([float(split[1]), float(split[2])])
        if line[:1] == "f":
            split = line.split()
            f_pos.append([int(split[1].split('/')[0]), int(split[2].split('/')[0]), int(split[3].split('/')[0])])
            f_uv.append([int(split[1].split('/')[1]), int(split[2].split('/')[1]), int(split[3].split('/')[1])])

    tri_bb_side = 0.04
    face_pts = {}
    for face_i in range(len(f_uv)):
        # Get UV points
        p1 = np.array(uv_pts[f_uv[face_i][0] - 1])
        p2 = np.array(uv_pts[f_uv[face_i][1] - 1])
        p3 = np.array(uv_pts[f_uv[face_i][2] - 1])
        # GET UV BBox
        ctr = (p1 + p2 + p3)/3.0
        ctr = ctr - tri_bb_side/2.0
        
        # Scale up to iterate on pixel grid
        ctr = np.floor(ctr * side)
        if(face_i % 5000 == 0):
            print(face_i)
        for i in range(int(ctr[0]), int(ctr[0]) + int(np.floor(tri_bb_side * side))):
            for j in range(int(ctr[1]), int(ctr[1]) + int(np.floor(tri_bb_side * side))):
                # Shrink points back to [0,1]
                pt_x = (float(i) - 0.5)/side
                pt_y = (float(j) - 0.5)/side
                pt = np.array([pt_x, pt_y])
                a = (pt - p1) / np.linalg.norm(pt - p1)
                b = (pt - p2) / np.linalg.norm(pt - p2)
                c = (pt - p3) / np.linalg.norm(pt - p3)
                ang_ab = np.arccos(np.dot(a, b))
                ang_bc = np.arccos(np.dot(b, c))
                ang_ca = np.arccos(np.dot(c, a))
                ang_sum = ang_ab + ang_bc + ang_ca
                
                if(ang_sum + 0.1 > 2.0 * np.pi and ang_sum - 0.1 < 2.0 * np.pi):
                    face_uv = np.array([p1,p2,p3])
                    baryPair = getBary(face_uv, np.array([pt_x, pt_y]))
                    if face_i in face_pts.keys():
                        face_pts[face_i].append(baryPair)
                    else:
                        face_pts[face_i] = [baryPair]
    
    # Get 3D points
    pts = []
    for face_i in range(len(f_uv)):
        if(i in face_pts.keys()):
            p1_3d = np.array(pos_pts[f_pos[face_i][0] - 1])
            p2_3d = np.array(pos_pts[f_pos[face_i][1] - 1])
            p3_3d = np.array(pos_pts[f_pos[face_i][2] - 1])
            
            face_3d = np.array([p1_3d,p2_3d,p3_3d])
            uvs = face_pts[face_i]
            for uv in range(len(uvs)):
                pt_3d = getPos(face_3d, uvs[uv])
                pts.append(pt_3d)
    return face_pts, pts

def main():
    filename_uv = 'uv_bary_dict'
    filename_3d = '3d_pts'

    face_pts, pts = make_bary_texels(100)
    save_obj(face_pts, filename_uv)
    save_obj(pts, filename_3d )
    
    uv = load_obj(filename_uv)
    pts = load_obj(filename_3d)
    print("PTS: ", pts[:10])
if __name__ == '__main__':
    main()