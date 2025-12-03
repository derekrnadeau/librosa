import librosa 
import numpy as np
import math
import netCDF4 as ncdf
import argparse
import sys
import soundfile as sf
from pathlib import Path
from datetime import datetime

### SOURCES:
# https://sofaconventions.org/data/amt-1.0.0/sofa/doc/SOFA%20specs%200.6.pdf
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
        sys.exit("InputError: Path to input file is invalid.")

    new_filename = (str(args.inputpath.stem) + 
        '_binaural_' + 
        str(datetime.now()).replace('.','').replace(' ', '_').replace(':','-') + 
        '.wav'
    )

    if (args.outputpath == None): 
        args.outputpath = Path(str(args.inputpath.parent) + '/' + new_filename)
        print("No output path provided. File will be created at:", args.outputpath)
    else:
        args.outputpath = Path(args.outputpath + '/' + new_filename)

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

    if args.crosstalkx in ["true", "True", "t", "1"]:
        args.crosstalkx = True
    elif args.crosstalkx in ["false", "False", "f", "0"]:
        args.crosstalkx = False
    else:
        print("Crosstalk cancellation input invalid. Defaulting to false.")
        args.crosstalkx = False


    return args

class Measurement: 
    def __init__(
        self, 
        IR, 
        position, 
        N: int
    ):
        self.N = N
        self.L = IR[0][0]
        self.R = IR[1][0]
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

    def get_IR(
        self, 
        target_azi: float, 
        target_ele: float
    ): 

        # find the closest angle measurement taken available in the file!
        # for all IR measurements measure distance between IR azi/ele and target azi/ele 
            # convert from spherical to cartesian coordinates
        # keep track of the measurement index with the smallest distance

        min_distance        = math.inf
        min_distance_index  = -1

        target_theta = np.deg2rad(target_azi)
        target_alpha = np.deg2rad(target_ele)

        # we assume distance is constant, and therefore a consistent unit sphere radius of 1 
        # theta is traditionally azimuth, phi is traditionally elevation 
        # alpha is angle from horizonatl plane, phi is angle from straight up

        # x = r × cos(alp) × cos(theta)
        # y = r × cos(alp) × sin(theta)
        # z = r × sin(alp)
        target_x = math.cos(target_alpha) * math.cos(target_theta)
        target_y = math.cos(target_alpha) * math.sin(target_theta)
        target_z = math.sin(target_alpha)
        
        # fancy vectorized array multiplication
        azi_array = np.array([m.azimuth   for m in self.measurements])
        ele_array = np.array([m.elevation for m in self.measurements])

        curr_theta = np.deg2rad(azi_array)
        curr_alpha = np.deg2rad(ele_array)

        curr_x = np.cos(curr_alpha) * np.cos(curr_theta)
        curr_y = np.cos(curr_alpha) * np.sin(curr_theta)
        curr_z = np.sin(curr_alpha)

        # distance between cartesian coords
        squared_distances = np.sqrt(
            ( target_x - curr_x ) ** 2 + 
            ( target_y - curr_y ) ** 2 + 
            ( target_z - curr_z ) ** 2 
        )

        min_distance_index = np.argmin(squared_distances)
        min_distance = squared_distances[min_distance_index]

        if min_distance == 0:
            message = "Exact match! Using IR with azi & ele:"
        else:
            message = "No exact match. Using IR with azi & ele:", 
            
        print(
                message,
                self.measurements[min_distance_index].azimuth, '-',
                self.measurements[min_distance_index].elevation, 
                "with distance:", min_distance
            )
        return self.measurements[min_distance_index]
    
    
# def cancel_crosstalk(signal):


parser = parser_setup()
args = handle_user_input(parser)
mySofa = SOFA(args.sofapath)
IR = mySofa.get_IR(args.azimuth, args.elevation)

y1, sr1 = librosa.load(args.inputpath.absolute()) 

yL = np.convolve(y1, IR.L, mode='full')
yR = np.convolve(y1, IR.R, mode='full')

max_amplitude = max(np.max(np.abs(yL)), np.max(np.abs(yR)))
yL = yL / max_amplitude
yR = yR / max_amplitude

stereo_data = np.column_stack((yL, yR))

sf.write(
    args.outputpath.absolute(),
    stereo_data, # Pass the combined 2D array
    mySofa.SR, 
    subtype='PCM_24',
    format='WAV'
)

