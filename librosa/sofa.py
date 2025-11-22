import librosa 
import numpy as np
import math
import netCDF4 as ncdf
import argparse
import sys
from pathlib import Path
from datetime import datetime

### SOURCES:
# https://unidata.github.io/netcdf4-python/
# https://www.mathworks.com/help/simulink/ref_extras/sphericaltocartesian.html
# https://docs.python.org/3/library/argparse.html

## TODO: enforce matching sample rates 
## TODO: integrate with librosa 
## TODO: write stereo wav file

def parser_setup():
    parser = argparse.ArgumentParser(
        prog="Script to load, read, and utilize SOFA files",
        description="",
    )
    parser.add_argument('-f',   '--sofapath',    help="(string)  The path to your SOFA file.")
    parser.add_argument('-i',   '--inputpath',   help="(string)  The path to your mono audio file.")
    parser.add_argument('-o',   '--outputpath',  help="(string)  The desired path of the output file. Defaults to the directory your source audio came from.")
    parser.add_argument('-ele', '--elevation',   help="(float)   In degrees, the desired perceived height of your audio.")
    parser.add_argument('-azi', '--azimuth',     help="(float)   In degrees, the desired perceived lateral position of your audio.")
    parser.add_argument('-xtc', '--crosstalkx',  help="(boolean) Set to true to apply crosstalk cancellation for binaural reporoduction over loudspeakers. Defaults to false.")
    return parser

def handle_user_input(parser):

    args = parser.parse_args()
    
    if args.crosstalkx in ["true", "True", "t", "1"]:
        args.crosstalkx = True
    elif args.crosstalkx in ["false", "False", "f", "0"]:
        args.crosstalkx = False
    else:
        print("Crosstalk cancellation input invalid. Defaulting to false.")
        args.crosstalkx = False


    try:
        if (not args.sofapath): raise ValueError
    except:
        parser.print_help()
        sys.exit("InputError: No SOFA file provided.")

    try: 
        args.sofapath = Path(args.sofapath)
        if (not args.sofapath.exists()): raise ValueError
    except:
        parser.print_help()
        sys.exit("InputError: Path to SOFA is invalid.")

    try:
        if (not args.inputpath): raise ValueError
    except:
        parser.print_help()
        sys.exit("InputError: No input file provided.")

    try: 
        args.inputpath = Path(args.inputpath)
        if (not args.inputpath.exists()): raise ValueError
    except:
        parser.print_help()
        sys.exit("InputError: Path to SOFA is invalid.")

    if (not args.outputpath): 
        args.outputpath = Path(
            str(args.inputpath.parent) + 
            '/' +
            str(args.inputpath.stem) + 
            '_binaural_' + 
            str(datetime.now()).replace('.','').replace(' ', '_').replace(':','-') + 
            '.wav'
        )
        print("No output path provided. File will be created at:", args.outputpath)

    try:
        if (not args.elevation): raise ValueError
    except:
        parser.print_help()
        sys.exit("InputError: No elevation provided.")

    try:
        if (not args.azimuth): raise ValueError
    except:
        parser.print_help()
        sys.exit("InputError: No azimuth provided.")

    try:
        args.elevation = float(args.elevation)
        assert isinstance(args.elevation, float)
    except:
        parser.print_help()
        sys.exit("InputError: 'elevation' is not a valid float.")

    try:
        args.azimuth = float(args.azimuth)
        assert isinstance(args.azimuth, float)
    except:
        parser.print_help()
        sys.exit("InputError: 'azimuth' is not a valid float.")

    return args

class Measurement: 
    def __init__(
        self, 
        IR, 
        position, 
        N: int
    ):
        self.N = N
        self.L = IR[0]
        self.R = IR[1]
        self.azimuth    = position[0]
        self.elevation  = position[1]
        self.distance   = position[2]

class SOFA:

    # M = number of measurements taken (combinations of azimuth and elevation)
    # C = coordinate system in 3D
    # I = singleton dimension, defines scalar value (one sampling rate, one listener position in C coords)
    # N = num samples per measurement 
    # R = num receivers (should always be two ears)
    # E = num emitters (should always be one speaker)

    def __init__(self, file_path):
        self.sofa = ncdf.Dataset(file_path, 'r', format='NETCDF4')   # load file
        assert len(self.sofa.variables['ReceiverPosition'][:]) == 2       # you need two ears!
        assert len(self.sofa.variables['EmitterPosition'][:])  == 1       # you need one speaker!

        self.SR = self.sofa.variables['Data.SamplingRate'][0]             # get sampling rate
        self.I = 1
        self.C = 3
        self.M = self.sofa.dimensions['M'].size
        self.N = self.sofa.dimensions['N'].size

        self.measurements = [
            Measurement(
                self.sofa.variables['Data.IR'][i], 
                self.sofa.variables['SourcePosition'][i], 
                self.N)
            for i in range(self.M)]
        
    def get_IR(
        self, 
        target_azi: float, 
        target_ele: float
    ): 

        # find the closest angle measurement taken available in the file!
        ### could definitely be optimized with more eloquent python
        ### I wasn't familiar with the math, so I left it more explicit 

        # in a loop of all IR measurements
        # measure distance between IR azi/ele and target azi/ele 
            # convert from spherical to cartesian coordinates
        # keep track of the measurement index with the smallest distance

        min_distance        = math.inf
        min_distance_index  = -1

        target_azi  = np.deg2rad(target_azi)
        target_ele  = np.deg2rad(target_ele)

        # we assume distance is constant, and therefore a consistent unit sphere radius of 1 
        # theta is traditionally azimuth, phi is traditionally elevation 

        # x = r × sin(phi) × cos(theta)
        # y = r × sin(phi) × sin(theta)
        # z = r × cos(phi)
        target_x = math.sin(target_ele) * math.cos(target_azi)
        target_y = math.sin(target_ele) * math.sin(target_azi)
        target_z = math.cos(target_ele)

        for i in range(self.M): 
            curr_azi = np.deg2rad(self.measurements[i].azimuth)
            curr_ele = np.deg2rad(self.measurements[i].elevation)
            
            curr_x = math.sin(curr_ele) * math.cos(curr_azi)
            curr_y = math.sin(curr_ele) * math.sin(curr_azi)
            curr_z = math.cos(curr_ele)

            # distances between two points in cartesian 3D 
            curr_distance = math.sqrt(
                math.pow(target_x - curr_x, 2) +
                math.pow(target_y - curr_y, 2) + 
                math.pow(target_z - curr_z, 2)
            )

            # update best fit if applicable
            if curr_distance < min_distance:
                min_distance = curr_distance
                min_distance_index = i
            
            # if ever a perfect match, return immediately
            if curr_distance == 0:
                print(
                    "Exact match! Using IR with azi & ele: ", 
                    self.measurements[min_distance_index].azimuth, 
                    ' - ',
                    self.measurements[min_distance_index].elevation
                )
                return self.measurements[min_distance_index]

        print(
            "No exact match. Using IR with azi & ele: ", 
            self.measurements[min_distance_index].azimuth, 
            ' - ',
            self.measurements[min_distance_index].elevation, 
            " with distance: ", min_distance
        )
        return self.measurements[min_distance_index]
    
    # <class 'netCDF4.Dataset'>
    # root group (NETCDF4 data model, file format HDF5):
    #     dimensions(sizes):        I(1), C(3), R(2), E(1), N(14400), M(50), S(0)
    #     variables(dimensions):    float64 ListenerPosition(I, C), 
    #                               float64 ReceiverPosition(R, C, I), 
    #                               float64 SourcePosition(M, C), 
    #                               float64 EmitterPosition(E, C, I), 
    #                               float64 ListenerUp(I, C), 
    #                               float64 ListenerView(I, C), 
    #                               float64 EmitterUp(E, C, I), 
    #                               float64 EmitterView(E, C, I), 
    #                               float64 Data.IR(M, R, E, N), //// 50 measurements, 2 receivers/ears, 1 emitter, n samples 
    #                               float64 Data.SamplingRate(I), 
    #                               float64 Data.Delay(I, R, E)
    #     Conventions:              SOFA
    #     Version:                  1.0
    #     SOFAConventions:          MultiSpeakerBRIR
    #     SOFAConventionsVersion:   0.3
    #     APIName:                  ARI SOFA API for Matlab/Octave
    #     APIVersion:               1.1.1
    #     ApplicationName:          MATLAB
    #     ApplicationVersion:       R2018a
    #     AuthorContact:            gavin.kearney@york.ac.uk
    #     Comment:                  50 source positions. KU100 subject. Measurement utlized Genelec 8030/8040 Loudspeakers and 3s swept sine technique. Reciever microphones were KU100 built in mics via MOTU UltraLite-mk3 Hybrid Preamps. Filters were free field equalized for the microphones responces. (approx.) Linear phase BRIRs.
    #     DataType:                 FIRE
    #     History:                  Measurement, Microphone Free Field Equalization, Trim
    #     License:                  Copyright 2018, University of York, Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
    #     Organization:             AudioLab, Department of Electronic Engineering, University of York, United Kingdom.
    #     References:               https://doi.org/10.3390/app8112029
    #     RoomType:                 reverberant
    #     Origin:                   Exponential Swept Sine Wave Measuremnt
    #     DateCreated:              2019-08-01 13:21:06
    #     DateModified:             2020-05-13 21:33:31
    #     Title:                    D1 BRIRs
    #     DatabaseName:             SADIE II
    #     ListenerShortName:        D1
    #     RoomDescription:          Acoustically Treated Listening Environment
    #     NCProperties:             version=1|netcdflibversion=4.6.1|hdf5libversion=1.8.12
    #     groups: 

parser = parser_setup()
args = handle_user_input(parser)
mySofa = SOFA(args.sofapath)
IR = mySofa.get_IR(args.azimuth, args.elevation)


y1, sr1 = librosa.load(args.inputpath.absolute(), sr=mySofa.SR)  ### works up to here
# yL, sr2 = librosa.load(IR.L, sr=mySofa.SR)  
# yR, sr2 = librosa.load(IR.R, sr=mySofa.SR)  

# if sr1 != sr2:
#     print("Warning: Sample rates differ. Resampling signal 2 to match signal 1.")
#     y1 = librosa.resample(y=y1, orig_sr=sr1, target_sr=sr2)

# convolved_signal_fft = librosa.fftconvolve(y1, yL, mode='full')

