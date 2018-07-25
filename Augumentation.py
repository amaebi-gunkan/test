# coding: utf-8  
import numpy as np
import SimpleITK as sitk
import os
import scipy
from scipy import ndimage as ndi

class Augmentor():
    """description of class

    @breif from Augmentation import Augmentor
    Augmentor class
    --SitkImage

    (update date)
    *20180725 : implement transform_label_image
              : deal with direction
    *20180719 : make
    """
    def __init__(self, SitkImage):
        
        #order : (x,y,z)
        self.shape = SitkImage.GetSize()
        self.spacing = SitkImage.GetSpacing()
        self.origin = SitkImage.GetOrigin()
        self.direction = SitkImage.GetDirection()
        
        self.args = self.shape + (sitk.sitkFloat32,)
        

    def warp_field2D(self, sigma=4.0, grid_spacing=32):

        # Initialize bspline transform and take margin for BSpline
        shape = (self.shape[0] + grid_spacing + self.shape[0]%grid_spacing, self.shape[1] + grid_spacing + self.shape[1]%grid_spacing)
        grid_num = [int(shape[0]/grid_spacing), int(shape[1]/grid_spacing)]

        args = shape + (sitk.sitkFloat32,)
        ref_image = sitk.Image(*args)
        tx = sitk.BSplineTransformInitializer(ref_image, [grid_num[0], grid_num[1]])

        # Initialize shift in control points:
        # mesh size = number of control points - spline order
        p = np.random.normal(0.0, sigma*self.spacing[0], (grid_num[0]+3, grid_num[1]+3, 2))

        # Anchor the edges of the image
        p[:, 0, :] = 0
        p[:, -1:, :] = 0
        p[0, :, :] = 0
        p[-1:, :, :] = 0


        # Set bspline transform parameters to the above shifts
        tx.SetParameters(p.flatten())

        # Compute deformation field
        displacement_filter = sitk.TransformToDisplacementFieldFilter()  
        displacement_filter.SetReferenceImage(ref_image)
        displacement_field = displacement_filter.Execute(tx)

        edge = (int((shape[0]-self.shape[0])/2), int((shape[1]-self.shape[1])/2))
        displacement_fieldO = displacement_field[edge[0]:edge[0] + self.shape[0],edge[1]:edge[1] + self.shape[1]]

        displacement_fieldO.SetSpacing(self.spacing)
        displacement_fieldO.SetOrigin(self.origin)
        displacement_fieldO.SetDirection(self.direction)
      
        return displacement_fieldO



    def warp_field3D(self, sigma=4.0, grid_spacing=32):

        # Initialize bspline transform and take margin for BSpline
        shape = (self.shape[0] + grid_spacing + self.shape[0]%grid_spacing, self.shape[1] + grid_spacing + self.shape[1]%grid_spacing, self.shape[2] + grid_spacing + self.shape[2]%grid_spacing)
        grid_num = [int(shape[0]/grid_spacing), int(shape[1]/grid_spacing), int(shape[2]/grid_spacing)]
  
        args = shape + (sitk.sitkFloat32,)
        ref_image = sitk.Image(*args)
        tx = sitk.BSplineTransformInitializer(ref_image, [grid_num[0], grid_num[1], grid_num[2]])

        # Initialize shift in control points:
        # mesh size = number of control points - spline order
        p = np.random.normal(0.0, sigma*self.spacing[0], (grid_num[0]+3, grid_num[1]+3, grid_num[2]+3, 3))

        # Anchor the edges of the image
        p[:, 0, :, :] = 0
        p[:, -1:, : ,:] = 0
        p[0, :, :, :] = 0
        p[-1:, :, :, :] = 0
        p[:, :, 0, :] = 0
        p[:, :, -1:, :] = 0

        # Set bspline transform parameters to the above shifts
        tx.SetParameters(p.flatten())

        # Compute deformation field
        displacement_filter = sitk.TransformToDisplacementFieldFilter()      
        displacement_filter.SetReferenceImage(ref_image)
        displacement_field = displacement_filter.Execute(tx)

        edge = (int((shape[0]-self.shape[0])/2), int((shape[1]-self.shape[1])/2), int((shape[2]-self.shape[2])/2))
        displacement_fieldO = displacement_field[edge[0]:edge[0] + self.shape[0],edge[1]:edge[1] + self.shape[1],edge[2]:edge[2] + self.shape[2]]

        displacement_fieldO.SetSpacing(self.spacing)
        displacement_fieldO.SetOrigin(self.origin)
        displacement_fieldO.SetDirection(self.direction)
      
        return displacement_fieldO



    def rotate_field3D(self, angle=5.0, translation=(0,0,0)):

        ref_image = sitk.Image(*self.args)
        rotationCenter = (self.shape[0]/2, self.shape[1]/2, self.shape[2]/2)
        rad = 2.0*np.pi/360.0*angle
        
        theta_x = rad * (np.random.random()*2-1)
        theta_y = rad * (np.random.random()*2-1)
        theta_z = rad * (np.random.random()*2-1)
        T = translation
        
        rigid_euler = sitk.Euler3DTransform(rotationCenter, theta_x, theta_y, theta_z, T)

        displacement_filter = sitk.TransformToDisplacementFieldFilter()
        displacement_filter.SetReferenceImage(ref_image)
        displacement_field = displacement_filter.Execute(rigid_euler)
        
        displacement_field.SetSpacing(self.spacing)
        displacement_field.SetOrigin(self.origin)
        displacement_field.SetDirection(self.direction)
        
        return displacement_field



    def transform_image(self, ImageI, displacement_field, interpolater = sitk.sitkLinear, default_value = -1000, out_type = sitk.sitkFloat32):
      
        fieldI = sitk.Image(displacement_field)
        tx = sitk.DisplacementFieldTransform(fieldI)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ImageI)
        resampler.SetInterpolator(interpolater)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetOutputPixelType(out_type)
        resampler.SetTransform(tx)

        ImageO = resampler.Execute(ImageI)
        
        return ImageO
    
    def transform_label_image(self, ImageI, displacement_field, label_num=1, interpolater = sitk.sitkLinear, default_value = 0,out_type = sitk.sitkUInt8):

        fieldI = sitk.Image(displacement_field)
        
        tx = sitk.DisplacementFieldTransform(fieldI)
        diftype = sitk.sitkFloat32
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ImageI)
        resampler.SetInterpolator(interpolater)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetOutputPixelType(diftype)
        resampler.SetTransform(tx)

        ArrayI = sitk.GetArrayFromImage(ImageI)
        ArrayO = np.array(ArrayI)
        dist_in = ndi.distance_transform_edt(np.where(ArrayO==1,1,0))
        dist_out = ndi.distance_transform_edt(np.where(ArrayO==1,0,1))
        
        dist = dist_out - dist_in
        dist[dist>=0] -= 0.5
        dist[dist<0] += 0.5
        ImageO = sitk.GetImageFromArray(dist)
        ImageO = resampler.Execute(ImageO)<0
        temp = sitk.GetArrayFromImage(ImageO)
        temp = np.array(temp).astype(np.uint8)
        ImageO = sitk.GetImageFromArray(temp)
        
        ImageO.SetSpacing(self.spacing)
        ImageO.SetOrigin(self.origin)
        ImageO.SetDirection(self.direction)
                
        if(label_num>1):
            for i in range(1,label_num+1):
                dist_in = ndi.distance_transform_edt(np.where(ArrayO==i,1,0))
                dist_out = ndi.distance_transform_edt(np.where(ArrayO==i,0,1))
                dist = dist_out - dist_in
                dist[dist>=0] -= 0.5
                dist[dist<0] += 0.5
                
                ImageD = sitk.GetImageFromArray(dist)
                ImageD.SetSpacing(self.spacing)
                ImageD.SetOrigin(self.origin)
                ImageD.SetDirection(self.direction)

                ImageD = resampler.Execute(ImageD)<0
                ImageO += ImageD*i
            
        return ImageO


    def save_displacement_field(self, displacement_field, OutputFileDir, OutputFileName):
        
        ArrayD = sitk.GetArrayFromImage(displacement_field)
        ArrayD = ArrayD.transpose()

        direction = ["x","y","z"]

        for i in range(len(ArrayD)):
            ImageD = sitk.GetImageFromArray(ArrayD[i])

            if not( os.path.isdir(OutputFileDir + "/" + direction[i] + "/")):
                os.mkdir(OutputFileDir + "/" + direction[i] + "/")
            sitk.WriteImage(ImageD,OutputFileDir + "/" + direction[i] + "/" + OutputFileName + ".mhd") 
