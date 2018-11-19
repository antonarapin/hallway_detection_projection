import numpy as np
from math import *

##################################################
####These functions are utility functions that we used during our project.
##################################################



def get_parallel(lines):
        angle_thresh = 0.07
        parallel_sets = []
        belong = [False for x in range(len(lines))]
        for lidx in range(len(lines)):
                if not belong[lidx]:
                        cur_set = []
                        cur_set.append(lidx)
                        belong[lidx]=True
                        v1 = [lines[lidx][1][0]-lines[lidx][0][0],lines[lidx][1][1]-lines[lidx][0][1]]
                        for l1 in range(len(lines)):
                                if not belong[l1]:
                                        v2 = [lines[l1][1][0]-lines[l1][0][0],lines[l1][1][1]-lines[l1][0][1]]
                                        c = (v1[0]*v2[0]+v1[1]*v2[1])/(sqrt(v1[0]**2 + v1[1]**2)*sqrt(v2[0]**2 + v2[1]**2))
                                        try:
                                                if acos(c)<angle_thresh:
                                                        cur_set.append(l1)
                                                        belong[l1]=True
                                        except:
                                                pass
                        parallel_sets.append(cur_set)
        return parallel_sets



def get_vertical_set_idx(pl, lines):
        vert_idxs = [0,0,0]
        vert_angle_off = [100,100,100]
        vert_thresh = 0.1
        for i in range(len(pl)):
                line = lines[pl[i][0]]
                v = [line[1][0]-line[0][0],line[1][1]-line[0][1]]
                angle = -1
                if v[0] == 0:
                        angle = 1.5708
                else:
                        angle = float(atan(float(v[1])/float(v[0])))
                diff = abs(1.5708 - angle)
                if diff<vert_thresh:
                        j = 0
                        stop = False
                        while not stop and j<len(vert_idxs):
                                if vert_angle_off[j]>diff:
                                        vert_angle_off[j]=diff
                                        vert_idxs[j] = i
                                        stop = True
                                j+=1
        if vert_angle_off[0] == 100:
                return -1

        m = max([len(pl[x]) for x in vert_idxs])
        for i in vert_idxs:
                if len(pl[i])==m:
                        return idx


def get_perp_line(idx, pl, lines):
        angle_sum = 0
        for l_idx in pl[idx]:
                line = lines[l_idx]
                v = [line[1][0]-line[0][0],line[1][1]-line[0][1]]
                if v[0]==0:
                        angle_sum+=1.5708
                else:
                        angle_sum+=float(atan(float(v[1])/float(v[0])))
        avg_inclination = float(float(angle_sum)/len(pl[idx]))
        perp_thresh = 0.05
        perp_idx = idx
        for line_set in pl:
                if not line_set == pl[idx]:
                        a_sum = 0
                        for p_idx in line_set:
                                line = lines[p_idx]
                                v = [line[1][0]-line[0][0],line[1][1]-line[0][1]]
                                if v[0] == 0:
                                        a_sum+=1.5708
                                else:
                                        a_sum+=float(atan(float(v[1])/float(v[0])))
                        a_incl = float(float(a_sum)/len(pl[idx]))
                        if (1.5708+perp_thresh)>avg_inclination and (1.5708-perp_thresh)<avg_inclination:
                                perp_idx = pl.index(line_set)
                                break
        return perp_idx


def get_max_points_line(pl):
        idx = 0
        max_p = 0
        for i in range(len(pl)):
                if len(pl[i])>max_p:
                        max_p = len(pl[i])
                        idx = i
        return i
