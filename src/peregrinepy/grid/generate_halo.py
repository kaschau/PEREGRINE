from numpy import s_

def generate_halo(blk):

   for var in ['x','y','z']:
   #face1
       fs0_ = s_[0,1:blk.nj+1,1:blk.nk+1]
       fs1_ = s_[1,1:blk.nj+1,1:blk.nk+1]
       fs2_ = s_[2,1:blk.nj+1,1:blk.nk+1]
       x = blk.array[var]
       x[fs0_] = 2.0*x[fs1_] - x[fs2_]

   #face2
       fs0_ = s_[-1,1:blk.nj+1,1:blk.nk+1]
       fs1_ = s_[-2,1:blk.nj+1,1:blk.nk+1]
       fs2_ = s_[-3,1:blk.nj+1,1:blk.nk+1]
       x = blk.array[var]
       x[fs0_] = 2.0*x[fs1_] - x[fs2_]

   #face3
       fs0_ = s_[1:blk.ni+1,0,1:blk.nk+1]
       fs1_ = s_[1:blk.ni+1,1,1:blk.nk+1]
       fs2_ = s_[1:blk.ni+1,2,1:blk.nk+1]
       x = blk.array[var]
       x[fs0_] = 2.0*x[fs1_] - x[fs2_]

   #face4
       fs0_ = s_[1:blk.ni+1,-1,1:blk.nk+1]
       fs1_ = s_[1:blk.ni+1,-2,1:blk.nk+1]
       fs2_ = s_[1:blk.ni+1,-3,1:blk.nk+1]
       x = blk.array[var]
       x[fs0_] = 2.0*x[fs1_] - x[fs2_]

   #face5
       fs0_ = s_[1:blk.ni+1,1:blk.nj+1,0]
       fs1_ = s_[1:blk.ni+1,1:blk.nj+1,1]
       fs2_ = s_[1:blk.ni+1,1:blk.nj+1,2]
       x = blk.array[var]
       x[fs0_] = 2.0*x[fs1_] - x[fs2_]

   #face6
       fs0_ = s_[1:blk.ni+1,1:blk.nj+1,-1]
       fs1_ = s_[1:blk.ni+1,1:blk.nj+1,-2]
       fs2_ = s_[1:blk.ni+1,1:blk.nj+1,-3]
       x = blk.array[var]
       x[fs0_] = 2.0*x[fs1_] - x[fs2_]

   #edge13
       fs10_ = s_[0,0,1:blk.nk+1]
       fs11_ = s_[1,0,1:blk.nk+1]
       fs12_ = s_[2,0,1:blk.nk+1]
       fs21_ = s_[0,1,1:blk.nk+1]
       fs22_ = s_[0,2,1:blk.nk+1]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )

   #edge14
       fs10_ = s_[0,-1,1:blk.nk+1]
       fs11_ = s_[1,-1,1:blk.nk+1]
       fs12_ = s_[2,-1,1:blk.nk+1]
       fs21_ = s_[0,-2,1:blk.nk+1]
       fs22_ = s_[0,-3,1:blk.nk+1]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )

   #edge15
       fs10_ = s_[0,1:blk.nj+1,0]
       fs11_ = s_[1,1:blk.nj+1,0]
       fs12_ = s_[2,1:blk.nj+1,0]
       fs21_ = s_[0,1:blk.nj+1,1]
       fs22_ = s_[0,1:blk.nj+1,2]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )

   #edge16
       fs10_ = s_[0,1:blk.nj+1,-1]
       fs11_ = s_[1,1:blk.nj+1,-1]
       fs12_ = s_[2,1:blk.nj+1,-1]
       fs21_ = s_[0,1:blk.nj+1,-2]
       fs22_ = s_[0,1:blk.nj+1,-3]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )

   #edge23
       fs10_ = s_[-1,0,1:blk.nk+1]
       fs11_ = s_[-2,0,1:blk.nk+1]
       fs12_ = s_[-3,0,1:blk.nk+1]
       fs21_ = s_[-1,1,1:blk.nk+1]
       fs22_ = s_[-1,2,1:blk.nk+1]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )

   #edge24
       fs10_ = s_[-1,-1,1:blk.nk+1]
       fs11_ = s_[-2,-1,1:blk.nk+1]
       fs12_ = s_[-3,-1,1:blk.nk+1]
       fs21_ = s_[-1,-2,1:blk.nk+1]
       fs22_ = s_[-1,-3,1:blk.nk+1]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )

   #edge25
       fs10_ = s_[-1,1:blk.nj+1,0]
       fs11_ = s_[-2,1:blk.nj+1,0]
       fs12_ = s_[-3,1:blk.nj+1,0]
       fs21_ = s_[-1,1:blk.nj+1,1]
       fs22_ = s_[-1,1:blk.nj+1,2]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )

   #edge26
       fs10_ = s_[-1,1:blk.nj+1,-1]
       fs11_ = s_[-2,1:blk.nj+1,-1]
       fs12_ = s_[-3,1:blk.nj+1,-1]
       fs21_ = s_[-1,1:blk.nj+1,-2]
       fs22_ = s_[-1,1:blk.nj+1,-3]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )

   #edge35
       fs10_ = s_[1:blk.ni+1,0,0]
       fs11_ = s_[1:blk.ni+1,1,0]
       fs12_ = s_[1:blk.ni+1,2,0]
       fs21_ = s_[1:blk.ni+1,0,1]
       fs22_ = s_[1:blk.ni+1,0,2]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )

   #edge36
       fs10_ = s_[1:blk.ni+1,0,-1]
       fs11_ = s_[1:blk.ni+1,1,-1]
       fs12_ = s_[1:blk.ni+1,2,-1]
       fs21_ = s_[1:blk.ni+1,0,-2]
       fs22_ = s_[1:blk.ni+1,0,-3]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )

   #edge45
       fs10_ = s_[1:blk.ni+1,-1,0]
       fs11_ = s_[1:blk.ni+1,-2,0]
       fs12_ = s_[1:blk.ni+1,-3,0]
       fs21_ = s_[1:blk.ni+1,-1,1]
       fs22_ = s_[1:blk.ni+1,-1,2]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )

   #edge46
       fs10_ = s_[1:blk.ni+1,-1,-1]
       fs11_ = s_[1:blk.ni+1,-2,-1]
       fs12_ = s_[1:blk.ni+1,-3,-1]
       fs21_ = s_[1:blk.ni+1,-1,-2]
       fs22_ = s_[1:blk.ni+1,-1,-3]

       x = blk.array[var]
       x[fs10_] = 0.5*( (2.0*x[fs11_] - x[fs12_]) +
                        (2.0*x[fs21_] - x[fs22_]) )


   #corner135
       fs10_ = s_[0,0,0]
       fs11_ = s_[1,0,0]
       fs12_ = s_[2,0,0]

       fs21_ = s_[0,1,0]
       fs22_ = s_[0,2,0]

       fs31_ = s_[0,0,1]
       fs32_ = s_[0,0,2]

       x = blk.array[var]
       x[fs10_] = (1.0/3.0)*( (2.0*x[fs11_] - x[fs12_]) +
                              (2.0*x[fs21_] - x[fs22_]) +
                              (2.0*x[fs31_] - x[fs32_]) )

   #corner145
       fs10_ = s_[0,-1,0]
       fs11_ = s_[1,-1,0]
       fs12_ = s_[2,-1,0]

       fs21_ = s_[0,-2,0]
       fs22_ = s_[0,-3,0]

       fs31_ = s_[0,-1,1]
       fs32_ = s_[0,-1,2]

       x = blk.array[var]
       x[fs10_] = (1.0/3.0)*( (2.0*x[fs11_] - x[fs12_]) +
                              (2.0*x[fs21_] - x[fs22_]) +
                              (2.0*x[fs31_] - x[fs32_]) )

   #corner136
       fs10_ = s_[0,0,-1]
       fs11_ = s_[1,0,-1]
       fs12_ = s_[2,0,-1]

       fs21_ = s_[0,1,-1]
       fs22_ = s_[0,2,-1]

       fs31_ = s_[0,0,-2]
       fs32_ = s_[0,0,-3]

       x = blk.array[var]
       x[fs10_] = (1.0/3.0)*( (2.0*x[fs11_] - x[fs12_]) +
                              (2.0*x[fs21_] - x[fs22_]) +
                              (2.0*x[fs31_] - x[fs32_]) )

   #corner146
       fs10_ = s_[0,-1,-1]
       fs11_ = s_[1,-1,-1]
       fs12_ = s_[2,-1,-1]

       fs21_ = s_[0,-2,-1]
       fs22_ = s_[0,-3,-1]

       fs31_ = s_[0,-1,-2]
       fs32_ = s_[0,-1,-3]

       x = blk.array[var]
       x[fs10_] = (1.0/3.0)*( (2.0*x[fs11_] - x[fs12_]) +
                              (2.0*x[fs21_] - x[fs22_]) +
                              (2.0*x[fs31_] - x[fs32_]) )

   #corner235
       fs10_ = s_[-1,0,0]
       fs11_ = s_[-2,0,0]
       fs12_ = s_[-3,0,0]

       fs21_ = s_[-1,1,0]
       fs22_ = s_[-1,2,0]

       fs31_ = s_[-1,0,1]
       fs32_ = s_[-1,0,2]

       x = blk.array[var]
       x[fs10_] = (1.0/3.0)*( (2.0*x[fs11_] - x[fs12_]) +
                              (2.0*x[fs21_] - x[fs22_]) +
                              (2.0*x[fs31_] - x[fs32_]) )

   #corner245
       fs10_ = s_[-1,-1,0]
       fs11_ = s_[-2,-1,0]
       fs12_ = s_[-3,-1,0]

       fs21_ = s_[-1,-2,0]
       fs22_ = s_[-1,-3,0]

       fs31_ = s_[-1,-1,1]
       fs32_ = s_[-1,-1,2]

       x = blk.array[var]
       x[fs10_] = (1.0/3.0)*( (2.0*x[fs11_] - x[fs12_]) +
                              (2.0*x[fs21_] - x[fs22_]) +
                              (2.0*x[fs31_] - x[fs32_]) )

   #corner236
       fs10_ = s_[-1,0,-1]
       fs11_ = s_[-2,0,-1]
       fs12_ = s_[-3,0,-1]

       fs21_ = s_[-1,1,-1]
       fs22_ = s_[-1,2,-1]

       fs31_ = s_[-1,0,-2]
       fs32_ = s_[-1,0,-3]

       x = blk.array[var]
       x[fs10_] = (1.0/3.0)*( (2.0*x[fs11_] - x[fs12_]) +
                              (2.0*x[fs21_] - x[fs22_]) +
                              (2.0*x[fs31_] - x[fs32_]) )

   #corner246
       fs10_ = s_[-1,-1,-1]
       fs11_ = s_[-2,-1,-1]
       fs12_ = s_[-3,-1,-1]

       fs21_ = s_[-1,-2,-1]
       fs22_ = s_[-1,-3,-1]

       fs31_ = s_[-1,-1,-2]
       fs32_ = s_[-1,-1,-3]

       x = blk.array[var]
       x[fs10_] = (1.0/3.0)*( (2.0*x[fs11_] - x[fs12_]) +
                              (2.0*x[fs21_] - x[fs22_]) +
                              (2.0*x[fs31_] - x[fs32_]) )
