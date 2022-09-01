import enum
import math
from math import floor
import os.path
import struct
import sys
import time
from tokenize import Double
import zmq
import typer
import argparse
import numpy as np

# Observer HK File Header Size !
file_header_size = 0  # 140 = 0x8C

print_bar_indent = '------------------------------'
print_cap_indent = '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
print_indent = ' '
print_only_unmatched = True
dummyDB = {}

class ValueType(enum.Enum):
    HEX = 1
    INT = 2
    FLOAT = 3

# ref : https://discuss.dizzycoding.com/reading-struct-in-python-from-created-struct-in-c/
# ref : https://sosomemo.tistory.com/59

# H1 은 수집하고 뿌려야되서 HK->fsw->tables->hk_cpy_tbl.c 참조
# 나머지는 각 app의 fsw->platform_inc->cs_msg.h 참조

# 구조체 bit 배열시 순서대로 lsb먼저들어가게됨
TestPack_t =  np.dtype([
    ("soh"                          , np.uint8   ,       ),
    ("ssoh"                         , np.uint8   ,       ),
    # uint8                           selectedSband       : 1;
    # uint8                           sbandStatus_P       : 1;
    # uint8                           sbandStatus_R       : 1;
    # uint8                           spare               : 5;
   
])

CFE_FS_Header_t =  np.dtype([
    ("ContentType"              , np.uint32   ,       ),    # /**< \brief Identifies the content type (='cFE1'=0x63464531)*/
    ("SubType"                  , np.uint32   ,       ),
    ("Length"                   , np.uint32   ,       ),
    ("SpacecraftID"             , np.uint32   ,       ),
    ("ProcessorID"              , np.uint32   ,       ),
    ("ApplicationID"            , np.uint32   ,       ),
    ("TimeSeconds"              , np.uint32   ,       ),
    ("TimeSubSeconds"           , np.uint32   ,       ),
    ("Description"              , np.uint8   , (32,)    ),
])

DS_FileHeader_t =  np.dtype([
    ("CloseSeconds"             , np.uint32   ,       ),
    ("CloseSubsecs"             , np.uint32   ,       ),
    ("FileTableIndex"           , np.uint16   ,       ),
    ("FileNameType"             , np.uint16   ,       ),
    # ("FileName"                 , np.uint8   , (64,)    ),
])

HKFILE_HEADER =  np.dtype([
    ("CFE_FS_Header_t"          , np.dtype(CFE_FS_Header_t) ,   ),
    ("DS_FileHeader_t"          , np.dtype(DS_FileHeader_t) ,   ),
    
])

#######################################################################################

EPS_FSW_HKPack_t =  np.dtype([
    ("soh"                          , np.uint8   ,       ),
    # uint8                           selectedSband       : 1;
    # uint8                           sbandStatus_P       : 1;
    # uint8                           sbandStatus_R       : 1;
    # uint8                           spare               : 5;
    
    ("operationMode"                , np.uint8   ,       ),
    ("operationSubMode"             , np.uint8   ,       ),
    ("rebootCount"                  , np.uint16   ,       ),
    ("nandFlashCapacity"            , np.uint8   ,       ),
    ("sdCardCapacity"               , np.uint8   ,       ),
])

CDHS_HKPack_t =  np.dtype([
    ("temperature"                  , np.int16      ,       ),
    ("edacCount"                    , np.uint16     ,       ),
])


EPS_SP_HKPack_t =  np.dtype([
    ("soh"                          , np.uint8   , (2,)      ),
    # uint16                           depStatus           : 6;               // EPS_SP_INDEX_LENGTH
    # uint16                           depStatus_GPIO      : 6;               // EPS_SP_INDEX_LENGTH
    # uint16                           spare               : 4;
    
    ("panelTemp_ST"                 , np.int16   , (4,)      ),
    ("panelTemp_UTT"                , np.int16   , (3,)      ),
    ("panelTemp_UBT"                , np.int16   , (3,)      ),
    ("antennaTemp"                  , np.int16   , (2,)      ),
    ("eocTemp"                      , np.int16   , (2,)      ),
  
])

EPS_P60_HKPack_t =  np.dtype([
    ("soh"                          , np.uint8   , (8,)      ),
#     uint32                          dockDeviceStatus    : 12; 8 4~
#     uint32                          acuDeviceStatus     : 14; ~4 8 2~
#     uint32                          heaterStatus        :  2;
#     uint32                          spare1              :  4;

#     uint8                           pduDeviceStatus     :  8;
    
    # 2022.09.01 --------------------------------------------------------------
    # uint16_t                        powerSwitchStatus         :   7;
    # uint16_t                        xbandSwitchStatus         :   2;
    # uint16_t                        pdhsSwitchStatus          :   2;
    # uint16_t                        polcubeSwitchStatus       :   2;
    # uint16_t                        spare6                    :   3;
    
    # uint8_t                         antXPHeaterSwitchStatus   :   2;
    # uint8_t                         antXMHeaterSwitchStatus   :   2;
    # uint8_t                         eocHeaterSwitchStatus     :   2;
    # uint8_t                         spare7                    :   2;
    


    ("SwitchCurrent"                , np.int16    , (7,)      ),
    ("SwitchVoltage"                , np.uint16   , (7,)      ),
    ("SwitchlatchupCount"           , np.uint16   , (7,)      ),
    ("docklatchupCount"             , np.uint16   , (4,)      ),
    ("acuCurrent"                   , np.int16    , (5,)      ),
    ("acuVoltage"                   , np.uint16   , (5,)      ),
    
    ("dockTemp"                     , np.int16   , (2,)      ),
    ("pduTemp"                      , np.int16   ,       ),
    ("acuTemp"                      , np.int16   , (3,)      ),
    ("bat_temperature"              , np.int16   , (8,)      ),
    
    ("vbatt"                        , np.uint16   ,       ),
    ("chargeCurrent"                , np.int16   ,      ),
    ("dischargeCurrent"             , np.int16   ,       ),
    ("dockBootcause"                , np.int8   ,       ),
    ("acuBootcause"                 , np.int8   ,       ),
    ("pduBootcause"                 , np.int8   ,       ),
])

CS_AX2150_HKPack_t =  np.dtype([
    ("soh"                          , np.uint8   ,       ),
    # uint8                           bootCause : 4;
    # uint8                           spare     : 4;
    
    ("temperature_board"            , np.int16   ,      ),
    ("temperature_pa"               , np.int16   ,     ),
    ("totalTxBytes"                 , np.uint32   ,      ),
    ("totalRxBytes"                 , np.uint32   ,       ),
    ("bootCount"                    , np.uint16   ,       ),
])


HKFILE_H1 =  np.dtype([
    ("EPS_FSW_HKPack_t"             , np.dtype(EPS_FSW_HKPack_t)      ,       ),
    ("CDHS_HKPack_t"                , np.dtype(CDHS_HKPack_t)       ,        ),
    ("EPS_SP_HKPack_t"              , np.dtype(EPS_SP_HKPack_t)       ,       ),
    ("EPS_P60_HKPack_t"             , np.dtype(EPS_P60_HKPack_t)       ,      ),
    ("CS_AX2150_HKPack_t"           , np.dtype(CS_AX2150_HKPack_t)       ,       ),
])

#######################################################################################

AC_HKPack_t =  np.dtype([
    ("soh"                          , np.uint8   ,  (10,)       ),
    # uint16                          l0_Status               : 7; 7~
    # uint16                          momentumHealth          : 5; ~1, 4~
    # uint16                          magSourceUsed           : 4; ~4

    # uint32                          timeValid               : 1; 1~
    # uint32                          refs_valid              : 1; ~1~
    # uint32                          attCtrlHealth           : 3; ~3~
    # uint32                          sttOperatingMode        : 3; ~3,
    # uint32                          torqueRodeMode1         : 4; 4~
    # uint32                          torqueRodeMode2         : 4; ~4,
    # uint32                          torqueRodeMode3         : 4; 4~
    # uint32                          torqueRodeFiringPack    : 6; ~4, 2~
    # uint32                          tableUploadStatus       : 5; ~5~
    # uint32                          spare1                  : 1; ~1

    # uint32                          sunPoint_state          : 3; 3~       // 4 -> 3
    # uint32                          attDetHealth            : 3; ~3~
    # uint32                          attCmdHealth            : 5; ~2, 3~
    # uint32                          attStatus               : 5; ~5,
    # uint32                          runLowRateTask          : 1; 1~
    # uint32                          magHealth               : 2; ~2~
    # uint32                          cssHealth               : 2; ~2~
    # uint32                          imuHealth               : 2; ~2~
    # uint32                          gpsHealth               : 6; ~1, 5~
    # uint16                          spare2                  : 3; ~3,      // 2 -> 3
    
    ("taiSeconds"                   , np.uint64   ,      ),
    ("positionWrt_ECI"              , np.double   , (3,)     ),
    ("velocityWrt_ECI"              , np.double   , (3,)     ),
    ("qBodyWrt_ECI"                 , np.int32   , (4,)     ),
    ("filteredSpeed_RPM"            , np.int16   , (3,)     ),
    ("totalMomentumMag"             , np.float32   ,      ),
    
])

AC_HKExtraPack_t =  np.dtype([
    ("l0_Status"                    , np.uint32   ,  (7,)       ),
    ("lastAcceptCmd"                , np.uint8   ,  (3,)       ),
    ("lastRejCmd"                   , np.uint8   ,  (3,)       ),
    ("inertia"                      , np.int32   ,  (3,)       ),
    ("badAttTimer"                  , np.uint32   ,       ),
    ("badRateTimer"                 , np.uint32   ,       ),
    ("cssInvalidCount"              , np.uint16   ,       ),
    ("imuInvalidCount"              , np.uint16   ,       ),
    ("imuReinitCount"               , np.uint16   ,       ),
    ("residual"                     , np.int32   , (3,)      ),
    ("bodyRate"                     , np.int32   , (3,)      ),
    ("gyroBiasEst"                  , np.int16   , (3,)      ),
    ("cmdQBodyWrt_ECI"              , np.int32   , (4,)      ),
    ("commandedSun"                 , np.int16   , (3,)      ),
    ("dragEst"                      , np.int16   , (3,)      ),
    ("motorFaultCount"              , np.uint8   , (3,)      ),
    ("sttMedianNoiseAllTrkBlks"     , np.uint8   ,    ),
    ("median_bckGnd"                , np.uint8   ,    ),
    ("detTimeoutCount"              , np.uint16   ,    ),
    ("numAttitudeStars"             , np.uint8   ,    ),
    ("eigenError"                   , np.uint32   ,    ),
    ("sunVectorBody"                , np.int16   ,  (3,)   ),
    ("magVectorBody"                , np.int16   ,  (3,)   ), # 2022.09.01 추가
    
    ("rawSunSensorData"             , np.uint16   ,  (12,)   ),
    ("rawMagnetometerData"          , np.uint16   ,  (9,)   ),
    ("imuAvgVector"                 , np.float32   ,  (3,)   ),
    ("imuAvgVectorFrame"            , np.uint8   ,    ),
    ("hrRunCount"                   , np.uint32   ,    ),
    ("hrTimeUsec"                   , np.uint32   ,    ),
    ("detTemp"                      , np.int8   ,    ),
    ("imuTemp"                      , np.int16   ,    ),
    ("motorTemp"                    , np.int16   , (3,)   ),
    ("digitalBus_V"                 , np.int16   ,  ),
    ("motorBus_V"                   , np.int16   ,  ),
    ("rodBus_v"                     , np.int16   ,  ),
    ("gpsCyclesSinceCRCData"        , np.uint32   ,  ),
    ("gpsCyclesSinceLatestData"     , np.uint32   ,  ),
    ("gpsLockCount"                 , np.uint16   ,  ),
    ("avgTimeTag"                   , np.uint32   ,  ),
    
])

HKFILE_H2 =  np.dtype([
    ("AC_HKPack_t"                  , np.dtype(AC_HKPack_t)      ,       ),
])

#######################################################################################

CS_EWC27_HKPack_t =  np.dtype([
    ("soh"                          , np.uint8   ,  (2,)       ),
    # uint16                          sourceOfLastStartup : 10;
    # uint16                          stateOfUnit         : 3;
    # uint16                          selfTestResult      : 3;
    
    ("temperature"                  , np.int8   ,      ),
    
])

HKFILE_H3 =  np.dtype([
    ("CS_EWC27_HKPack_t"            , np.dtype(CS_EWC27_HKPack_t)      ,       ),
])

#######################################################################################

PC_PDHS_HKPack_t =  np.dtype([
    ("errorStatus"                  , np.uint8   ,      ),
    ("receivedCommandCount"         , np.uint32   ,      ),
    ("receivedCommandErrorCount"    , np.uint32   ,      ),
    ("temperature"                  , np.int16   ,     ),
])

HKFILE_H4 =  np.dtype([
    ("PC_PDHS_HKPack_t"                , np.dtype(PC_PDHS_HKPack_t)      ,       ),
])

#######################################################################################

PC_PolCube_SOH_t =  np.dtype([
    ("boardstatus"                  , np.uint8   ,   ),
    # uint16   cameraId                : 1;    // Board ID: 0 - PolCube1, 1 - PolCube2
    # uint16   binningStatus           : 1;    // Binning Status: 0 - Stop, 1 - Run 
    # uint16   ldoStatus               : 1;    // LDO Status: 0 - Off, 1 - On
    # uint16   sensorSyncStatus        : 1;    // Sensor Sync Status: 0 - NG, 1 - OK
    # uint16   reserved                : 4;
])

PC_PolCube_HK_t =  np.dtype([
    ("currentTime"                      , np.uint32   ,     ),
    ("ldoBoardTemp"                     , np.uint16   ,     ),
    ("powerBoardTemp"                   , np.uint16   ,     ),
    ("ScifBoardTemp"                    , np.uint16   ,     ),
    ("FpgaTemp"                         , np.uint16   ,     ),
    ("SensorTemp"                       , np.uint16   ,     ),
    
])

HKFILE_H5 =  np.dtype([
    ("PC_PolCube_SOH_t"                , np.dtype(PC_PolCube_SOH_t)      ,       ),
    ("PC_PolCube_HK_t"                 , np.dtype(PC_PolCube_HK_t)      ,       ),
])

#######################################################################################

HKFILE_H6 =  np.dtype([
    ("AC_HKExtraPack_t"                , np.dtype(AC_HKExtraPack_t)      ,       ),
])

#######################################################################################

class HKFILE_H1_c:
    family = [  'EPS_FSW_HKPack_t', 
                'CDHS_HKPack_t', 
                'EPS_SP_HKPack_t', 
                'EPS_P60_HKPack_t',
                'CS_AX2150_HKPack_t']
    
    def __init__(self, bytes_of_values):
        header = np.frombuffer(bytes_of_values, dtype=HKFILE_H1)
        for x in self.family:
            self.buf = []
            self.buf = header[str(x)]
            self.switch(x)

    def switch(self, arg):
        self.case_name = "case_" + str(arg)
        self.case = getattr(self, self.case_name, lambda:"default")
        return self.case()
    
    def case_TestPack_t(self):
        print(f'{print_bar_indent}   [TestPack_t]')
        struct_data = np.frombuffer(self.buf, dtype=TestPack_t)
        temp = int(struct_data["soh"])
        print(f' soh    : {temp}')
    
    def case_EPS_FSW_HKPack_t(self):
        print(f'{print_bar_indent}   [EPS_FSW_HKPack_t]')
        struct_data = np.frombuffer(self.buf, dtype=EPS_FSW_HKPack_t)
        
        temp = int(struct_data["soh"])
        selectedSband = temp & 0x01
        sbandStatus_P = (temp & 0x02) >> 1
        sbandStatus_R = (temp & 0x04) >> 2
        spare = (temp & 0xF8) >> 3
        
        PrintAndCheck(1, 1, f'selectedSband', selectedSband, ValueType.HEX, 1)
        PrintAndCheck(1, 1, f'sbandStatus_P', sbandStatus_P, ValueType.HEX, 1)
        PrintAndCheck(1, 1, f'sbandStatus_R', sbandStatus_R, ValueType.HEX, 1)
        PrintAndCheck(1, 1, f'spare', spare, ValueType.HEX, 2)
        
        PrintAndCheck(1, 1, f'operationMode', int(struct_data["operationMode"]), ValueType.INT)
        PrintAndCheck(1, 1, f'operationSubMode', int(struct_data["operationSubMode"]), ValueType.INT)
        PrintAndCheck(1, 1, f'rebootCount', int(struct_data["rebootCount"]), ValueType.INT)
        PrintAndCheck(1, 1, f'nandFlashCapacity', int(struct_data["nandFlashCapacity"]), ValueType.INT)
        PrintAndCheck(1, 1, f'sdCardCapacity', int(struct_data["sdCardCapacity"]), ValueType.INT)
        
        
        print('')
    
    def case_CDHS_HKPack_t(self):
        print(f'{print_bar_indent}   [CDHS_HKPack_t]')
        struct_data = np.frombuffer(self.buf, dtype=CDHS_HKPack_t)
        
        PrintAndCheck(1, 2, f'temperature', int(struct_data["temperature"]), ValueType.INT)
        PrintAndCheck(1, 2, f'edacCount', int(struct_data["edacCount"]), ValueType.INT)
        
        print('')
        
    def case_EPS_SP_HKPack_t(self):
        print(f'{print_bar_indent}   [EPS_SP_HKPack_t]')
        struct_data = np.frombuffer(self.buf, dtype=EPS_SP_HKPack_t)
        
        #print(f'-----------3 {struct_data["soh"]}')
        # $ -----------3 [[50 51]]
        #print(f'-----------4 {struct_data["soh"][0][0]}')
        # $ -----------4 50
        
        temp = int(struct_data["soh"][0][0])
        temp += (int(struct_data["soh"][0][1]) << 8)
        
        depStatus = temp & 0b0000000000111111
        depStatus_GPIO = (temp & 0b0000111111000000) >> 6
        spare = (temp & 0b1111000000000000) >> 12
        
        PrintAndCheck(1, 3, f'depStatus', depStatus, ValueType.HEX, 2)
        PrintAndCheck(1, 3, f'depStatus_GPIO', depStatus_GPIO, ValueType.HEX, 2)
        PrintAndCheck(1, 3, f'spare', spare, ValueType.HEX, 2)
        
        label = "panelTemp_ST"
        for i in range(0, len(struct_data[label][0])):
            PrintAndCheck(1, 3, f'{label}_{i}', int(struct_data[label][0][i]), ValueType.INT)
        
        label = "panelTemp_UTT"
        for i in range(0, len(struct_data[label][0])):
            PrintAndCheck(1, 3, f'{label}_{i}', int(struct_data[label][0][i]), ValueType.INT)
                        
        label = "panelTemp_UBT"
        for i in range(0, len(struct_data[label][0])):
            PrintAndCheck(1, 3, f'{label}_{i}', int(struct_data[label][0][i]), ValueType.INT)
            
        label = "antennaTemp"
        for i in range(0, len(struct_data[label][0])):
            PrintAndCheck(1, 3, f'{label}_{i}', int(struct_data[label][0][i]), ValueType.INT)
            
        label = "eocTemp"
        for i in range(0, len(struct_data[label][0])):
            PrintAndCheck(1, 3, f'{label}_{i}', int(struct_data[label][0][i]), ValueType.INT)
       
        print('')
        
    def case_EPS_P60_HKPack_t(self):
        print(f'{print_bar_indent}   [EPS_P60_HKPack_t]')
        struct_data = np.frombuffer(self.buf, dtype=EPS_P60_HKPack_t)
        
        #print(f' ------ {struct_data["soh"][0]}')
        #print(f' ------ {struct_data["soh"][0][0]}')
        #print(f' ------ {struct_data["soh"][0][1]}')
        
        dockDeviceStatus = int(struct_data["soh"][0][0]) | ((int(struct_data["soh"][0][1]) & 0x0F) << 8)
        acuDeviceStatus = (int(struct_data["soh"][0][2]) << 4) | ((int(struct_data["soh"][0][1]) & 0xF0) >> 4) | ((int(struct_data["soh"][0][3]) & 0x03) << 12) 
        heaterStatus = ((int(struct_data["soh"][0][3]) & 0x0C) >> 2) 
        spare1 = ((int(struct_data["soh"][0][3]) & 0xF0) >> 4) 
        
        pduDeviceStatus = int(struct_data["soh"][0][4])
        
        powerSwitchStatus = int(struct_data["soh"][0][5]) | ((int(struct_data["soh"][0][6]) & 0x03) << 8)
        xbandSwitchStatus = (int(struct_data["soh"][0][6]) & 0x0C) >> 2
        pdhsSwitchStatus = (int(struct_data["soh"][0][6]) & 0x30) >> 4
        polcubeSwitchStatus = (int(struct_data["soh"][0][6]) & 0xC0) >> 6
        spare6 = (int(struct_data["soh"][0][6]) & 0xC0) >> 6
        
        
        powerSwitchStatus = (int(struct_data["soh"][0][5]) & 0x7F)
        xbandSwitchStatus = ((int(struct_data["soh"][0][5]) & 0x80) >> 7) | ((int(struct_data["soh"][0][6]) & 0x01) << 1)
        pdhsSwitchStatus = (int(struct_data["soh"][0][6]) & 0x06) >> 1
        polcubeSwitchStatus = (int(struct_data["soh"][0][6]) & 0x18) >> 3
        spare6 = (int(struct_data["soh"][0][6]) & 0xE0) >> 5
        
        antXPHeaterSwitchStatus = (int(struct_data["soh"][0][7]) & 0x03)
        antXMHeaterSwitchStatus = (int(struct_data["soh"][0][7]) & 0x0C) >> 2
        eocHeaterSwitchStatus = (int(struct_data["soh"][0][7]) & 0x30) >> 4
        spare7 = (int(struct_data["soh"][0][7]) & 0xC0) >> 6
        
        PrintAndCheck(1, 4, f'dockDeviceStatus', dockDeviceStatus, ValueType.HEX, 3)
        PrintAndCheck(1, 4, f'acuDeviceStatus', acuDeviceStatus, ValueType.HEX, 4)
        PrintAndCheck(1, 4, f'heaterStatus', heaterStatus, ValueType.HEX, 1)
        PrintAndCheck(1, 4, f'spare1', spare1, ValueType.HEX, 1)
        
        PrintAndCheck(1, 4, f'pduDeviceStatus', pduDeviceStatus, ValueType.HEX, 2)
        
        PrintAndCheck(1, 4, f'powerSwitchStatus', powerSwitchStatus, ValueType.HEX, 2)
        PrintAndCheck(1, 4, f'xbandSwitchStatus', xbandSwitchStatus, ValueType.HEX, 1)
        PrintAndCheck(1, 4, f'pdhsSwitchStatus', pdhsSwitchStatus, ValueType.HEX, 1)
        PrintAndCheck(1, 4, f'polcubeSwitchStatus', polcubeSwitchStatus, ValueType.HEX, 1)
        PrintAndCheck(1, 4, f'spare6', spare6, ValueType.HEX, 1)
        
        PrintAndCheck(1, 4, f'antXPHeaterSwitchStatus', antXPHeaterSwitchStatus, ValueType.HEX, 1)
        PrintAndCheck(1, 4, f'antXMHeaterSwitchStatus', antXMHeaterSwitchStatus, ValueType.HEX, 1)
        PrintAndCheck(1, 4, f'eocHeaterSwitchStatus', eocHeaterSwitchStatus, ValueType.HEX, 1)
        PrintAndCheck(1, 4, f'spare7', spare7, ValueType.HEX, 1)

        
        for value in range(0, len(struct_data["SwitchCurrent"][0])):
            PrintAndCheck(1, 4, f'SwitchCurrent[{value}]', int(struct_data["SwitchCurrent"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["SwitchVoltage"][0])):
            PrintAndCheck(1, 4, f'SwitchCurrent[{value}]', int(struct_data["SwitchCurrent"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["SwitchlatchupCount"][0])):
            PrintAndCheck(1, 4, f'SwitchlatchupCount[{value}]', int(struct_data["SwitchlatchupCount"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["docklatchupCount"][0])):
            PrintAndCheck(1, 4, f'docklatchupCount[{value}]', int(struct_data["docklatchupCount"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["acuCurrent"][0])):
            PrintAndCheck(1, 4, f'acuCurrent[{value}]', int(struct_data["acuCurrent"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["acuVoltage"][0])):
            PrintAndCheck(1, 4, f'acuVoltage[{value}]', int(struct_data["acuVoltage"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["dockTemp"][0])):
            PrintAndCheck(1, 4, f'dockTemp[{value}]', int(struct_data["dockTemp"][0][value]), ValueType.INT)
            
        PrintAndCheck(1, 4, f'pduTemp', int(struct_data["pduTemp"]), ValueType.INT)
            
        for value in range(0, len(struct_data["acuTemp"][0])):
            PrintAndCheck(1, 4, f'acuTemp[{value}]', int(struct_data["acuTemp"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["bat_temperature"][0])):
            PrintAndCheck(1, 4, f'bat_temperature[{value}]', int(struct_data["bat_temperature"][0][value]), ValueType.INT)
            
        PrintAndCheck(1, 4, f'vbatt', int(struct_data["vbatt"]), ValueType.INT)
        PrintAndCheck(1, 4, f'chargeCurrent', int(struct_data["chargeCurrent"]), ValueType.INT)
        PrintAndCheck(1, 4, f'dischargeCurrent', int(struct_data["dischargeCurrent"]), ValueType.INT)
        PrintAndCheck(1, 4, f'dockBootcause', int(struct_data["dockBootcause"]), ValueType.INT)
        PrintAndCheck(1, 4, f'acuBootcause', int(struct_data["acuBootcause"]), ValueType.INT)
        PrintAndCheck(1, 4, f'pduBootcause', int(struct_data["pduBootcause"]), ValueType.INT)
        
        print('')
        
    def case_CS_AX2150_HKPack_t(self):
        print(f'{print_bar_indent}   [CS_AX2150_HKPack_t]')
        struct_data = np.frombuffer(self.buf, dtype=CS_AX2150_HKPack_t)
    
        temp = int(struct_data["soh"])
        
        bootCause = temp & 0x0F
        spare = (temp & 0xF0) >> 4
        
        PrintAndCheck(1, 5, f'bootCause', bootCause, ValueType.HEX, 1)
        PrintAndCheck(1, 5, f'spare', spare, ValueType.HEX, 1)
        PrintAndCheck(1, 5, f'temperature_board', int(struct_data["temperature_board"]), ValueType.INT)
        PrintAndCheck(1, 5, f'temperature_pa', int(struct_data["temperature_pa"]), ValueType.INT)
        PrintAndCheck(1, 5, f'totalTxBytes', int(struct_data["totalTxBytes"]), ValueType.HEX, 8)
        PrintAndCheck(1, 5, f'totalRxBytes', int(struct_data["totalRxBytes"]), ValueType.HEX, 8)
        PrintAndCheck(1, 5, f'bootCount', int(struct_data["bootCount"]), ValueType.HEX, 4)
        
class HKFILE_H2_c:
    family = [  'AC_HKPack_t', 
                ]
    
    def __init__(self, bytes_of_values):
        header = np.frombuffer(bytes_of_values, dtype=HKFILE_H2)
        for x in self.family:
            self.buf = []
            self.buf = header[str(x)]
            self.switch(x)

    def switch(self, arg):
        self.case_name = "case_" + str(arg)
        self.case = getattr(self, self.case_name, lambda:"default")
        return self.case()
 
    def case_AC_HKPack_t(self):
        print(f'{print_bar_indent}   [AC_HKPack_t]')
        struct_data = np.frombuffer(self.buf, dtype=AC_HKPack_t)
        
        l0_Status = int(struct_data["soh"][0][0]) & 0b01111111
        momentumHealth = ((int(struct_data["soh"][0][0]) & 0x80) >> 7) | ((int(struct_data["soh"][0][1]) & 0x0F) << 1)
        magSourceUsed = ((int(struct_data["soh"][0][1]) & 0xF0) >> 4)
        
        timeValid = (int(struct_data["soh"][0][2]) & 0x01)
        refs_valid = (int(struct_data["soh"][0][2]) & 0x02)
        attCtrlHealth = (int(struct_data["soh"][0][2]) & 0x1C) >> 2
        sttOperatingMode = (int(struct_data["soh"][0][2]) & 0xE0) >> 5
        
        torqueRodeMode1 = int(struct_data["soh"][0][3]) & 0x0F
        torqueRodeMode2 = (int(struct_data["soh"][0][3]) & 0xF0) >> 4
        torqueRodeMode3 = (int(struct_data["soh"][0][4]) & 0x0F)
        torqueRodeFiringPack = ((int(struct_data["soh"][0][4]) & 0xF0) >> 4) | ((int(struct_data["soh"][0][5]) & 0x03) << 4)
        tableUploadStatus = ((int(struct_data["soh"][0][5]) & 0x7C) >> 2)
        spare1 = ((int(struct_data["soh"][0][5]) & 0x80) >> 7)
        
        sunPoint_state = (int(struct_data["soh"][0][6]) & 0x07)
        attDetHealth = (int(struct_data["soh"][0][6]) & 0x38) >> 3
        attCmdHealth = ((int(struct_data["soh"][0][6]) & 0xC0) >> 6) |  ((int(struct_data["soh"][0][7]) & 0x07) << 2)
        attStatus = ((int(struct_data["soh"][0][7]) & 0xF8) >> 3)
        runLowRateTask = (int(struct_data["soh"][0][8]) & 0x01)
        
        magHealth = ((int(struct_data["soh"][0][8]) & 0x06) >> 1)
        cssHealth = ((int(struct_data["soh"][0][8]) & 0x18) >> 3)
        imuHealth = ((int(struct_data["soh"][0][8]) & 0x60) >> 5) 
        gpsHealth = ((int(struct_data["soh"][0][8]) & 0x80) >> 7) | ((int(struct_data["soh"][0][9]) & 0x1F) << 1)
        spare2 = (int(struct_data["soh"][0][9]) & 0xE0) >> 5
        
        PrintAndCheck(2, 1, f'l0_Status', eval(f'l0_Status'), ValueType.HEX, 2)
        PrintAndCheck(2, 1, f'momentumHealth', momentumHealth, ValueType.HEX, 2)
        PrintAndCheck(2, 1, f'magSourceUsed', magSourceUsed, ValueType.HEX, 1)
        
        PrintAndCheck(2, 1, f'timeValid', timeValid, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'refs_valid', refs_valid, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'attCtrlHealth', attCtrlHealth, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'sttOperatingMode', sttOperatingMode, ValueType.HEX, 1)
        
        PrintAndCheck(2, 1, f'torqueRodeMode1', torqueRodeMode1, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'torqueRodeMode2', torqueRodeMode2, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'torqueRodeMode3', torqueRodeMode3, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'torqueRodeFiringPack', torqueRodeFiringPack, ValueType.HEX, 2)
        PrintAndCheck(2, 1, f'tableUploadStatus', tableUploadStatus, ValueType.HEX, 2)
        PrintAndCheck(2, 1, f'spare1', spare1, ValueType.HEX, 1)
        
        PrintAndCheck(2, 1, f'sunPoint_state', sunPoint_state, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'attDetHealth', attDetHealth, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'attCmdHealth', attCmdHealth, ValueType.HEX, 2)
        PrintAndCheck(2, 1, f'attStatus', attStatus, ValueType.HEX, 2)
        PrintAndCheck(2, 1, f'runLowRateTask', runLowRateTask, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'magHealth', magHealth, ValueType.HEX, 1)
        
        PrintAndCheck(2, 1, f'cssHealth', cssHealth, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'imuHealth', imuHealth, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'gpsHealth', gpsHealth, ValueType.HEX, 2)
        PrintAndCheck(2, 1, f'spare2', spare2, ValueType.HEX, 1)
        PrintAndCheck(2, 1, f'taiSeconds', int(struct_data["taiSeconds"]), ValueType.INT)
        
        for value in range(0, len(struct_data["positionWrt_ECI"][0])):
            #print(f' positionWrt_ECI[%d]                 : %.3lf' % (value, float(struct_data["positionWrt_ECI"][0][value])))
            PrintAndCheck(2, 1, f'positionWrt_ECI[{value}]', float(struct_data["positionWrt_ECI"][0][value]), ValueType.FLOAT)
        
        for value in range(0, len(struct_data["velocityWrt_ECI"][0])):
            #print(f' velocityWrt_ECI[%d]                 : %.3lf' % (value, float(struct_data["velocityWrt_ECI"][0][value])))
            PrintAndCheck(2, 1, f'velocityWrt_ECI[{value}]', float(struct_data["velocityWrt_ECI"][0][value]), ValueType.FLOAT)
                        
        for value in range(0, len(struct_data["qBodyWrt_ECI"][0])):
            PrintAndCheck(2, 1, f'qBodyWrt_ECI[{value}]',int(struct_data["qBodyWrt_ECI"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["filteredSpeed_RPM"][0])):
            PrintAndCheck(2, 1, f'filteredSpeed_RPM[{value}]',int(struct_data["filteredSpeed_RPM"][0][value]), ValueType.INT)
            
        #print(f' totalMomentumMag                   : %.3lf' % float(struct_data["totalMomentumMag"]))
        PrintAndCheck(2, 1, f'totalMomentumMag', float(struct_data["totalMomentumMag"]), ValueType.FLOAT)
        
        print('')
    
class HKFILE_H3_c:
    family = [  'CS_EWC27_HKPack_t', 
                ]
    
    def __init__(self, bytes_of_values):
        header = np.frombuffer(bytes_of_values, dtype=HKFILE_H3)
        for x in self.family:
            self.buf = []
            self.buf = header[str(x)]
            self.switch(x)

    def switch(self, arg):
        self.case_name = "case_" + str(arg)
        self.case = getattr(self, self.case_name, lambda:"default")
        return self.case()
 
    def case_CS_EWC27_HKPack_t(self):
        print(f'{print_bar_indent}   [CS_EWC27_HKPack_t]')
        struct_data = np.frombuffer(self.buf, dtype=CS_EWC27_HKPack_t)
        
        sourceOfLastStartup =  (int(struct_data["soh"][0][0]) & 0xFF) | ((int(struct_data["soh"][0][1]) & 0x02) << 8)
        stateOfUnit =  ((int(struct_data["soh"][0][1]) & 0x1C) >> 2)
        selfTestResult =  ((int(struct_data["soh"][0][1]) & 0xE0) >> 5)
 
        PrintAndCheck(3, 1, f'sourceOfLastStartup', sourceOfLastStartup, ValueType.HEX, 3)
        PrintAndCheck(3, 1, f'stateOfUnit', stateOfUnit, ValueType.HEX, 1)
        PrintAndCheck(3, 1, f'selfTestResult', selfTestResult, ValueType.HEX, 1)
        PrintAndCheck(3, 1, f'temperature', int(struct_data["temperature"]), ValueType.INT)

        print('')

class HKFILE_H4_c:
    family = [  'PC_PDHS_HKPack_t', 
                ]
    
    def __init__(self, bytes_of_values):
        header = np.frombuffer(bytes_of_values, dtype=HKFILE_H4)
        for x in self.family:
            self.buf = []
            self.buf = header[str(x)]
            self.switch(x)

    def switch(self, arg):
        self.case_name = "case_" + str(arg)
        self.case = getattr(self, self.case_name, lambda:"default")
        return self.case()
 
    def case_PC_PDHS_HKPack_t(self):
        print(f'{print_bar_indent}   [PC_PDHS_HKPack_t]')
        struct_data = np.frombuffer(self.buf, dtype=PC_PDHS_HKPack_t)
       
        PrintAndCheck(4, 1, f'errorStatus', int(struct_data["errorStatus"]), ValueType.INT)
        PrintAndCheck(4, 1, f'temperature', int(struct_data["temperature"]), ValueType.INT)
        PrintAndCheck(4, 1, f'receivedCommandCount', int(struct_data["receivedCommandCount"]), ValueType.INT)
        PrintAndCheck(4, 1, f'receivedCommandErrorCount', int(struct_data["receivedCommandErrorCount"]), ValueType.INT)

        print('')            
        
class HKFILE_H5_c:
    family = [  'PC_PolCube_SOH_t', 
                'PC_PolCube_HK_t',
                ]
    
    def __init__(self, bytes_of_values):
        header = np.frombuffer(bytes_of_values, dtype=HKFILE_H5)
        for x in self.family:
            self.buf = []
            self.buf = header[str(x)]
            self.switch(x)

    def switch(self, arg):
        self.case_name = "case_" + str(arg)
        self.case = getattr(self, self.case_name, lambda:"default")
        return self.case()
 
    def case_PC_PolCube_SOH_t(self):
        print(f'{print_bar_indent}   [PC_PolCube_SOH_t]')
        struct_data = np.frombuffer(self.buf, dtype=PC_PolCube_SOH_t)
       
        cameraId =  (int(struct_data["boardstatus"]) & 0x01)
        binningStatus =  (int(struct_data["boardstatus"]) & 0x02) >> 1
        ldoStatus =  (int(struct_data["boardstatus"]) & 0x04) >> 2
        sensorSyncStatus =  (int(struct_data["boardstatus"]) & 0x08) >> 3
        reserved =  ((int(struct_data["boardstatus"]) & 0xF0) >> 4)
        
        PrintAndCheck(5, 1, f'cameraId', cameraId, ValueType.HEX, 1)
        PrintAndCheck(5, 1, f'binningStatus', binningStatus, ValueType.HEX, 1)
        PrintAndCheck(5, 1, f'ldoStatus', ldoStatus, ValueType.HEX, 1)
        PrintAndCheck(5, 1, f'sensorSyncStatus', sensorSyncStatus, ValueType.HEX, 1)
        PrintAndCheck(5, 1, f'reserved', reserved, ValueType.HEX, 3)

        print('')            
    
    def case_PC_PolCube_HK_t(self):
        print(f'{print_bar_indent}   [PC_PolCube_HK_t]')
        struct_data = np.frombuffer(self.buf, dtype=PC_PolCube_HK_t)
       
        PrintAndCheck(5, 2, f'currentTime', int(struct_data["currentTime"]), ValueType.INT)
        PrintAndCheck(5, 2, f'ldoBoardTemp', int(struct_data["ldoBoardTemp"]), ValueType.INT)
        PrintAndCheck(5, 2, f'powerBoardTemp', int(struct_data["powerBoardTemp"]), ValueType.INT)
        PrintAndCheck(5, 2, f'ScifBoardTemp', int(struct_data["ScifBoardTemp"]), ValueType.INT)
        PrintAndCheck(5, 2, f'FpgaTemp', int(struct_data["FpgaTemp"]), ValueType.INT)
        PrintAndCheck(5, 2, f'SensorTemp', int(struct_data["SensorTemp"]), ValueType.INT)

        print('')             
        
class HKFILE_H6_c:
    family = [  'AC_HKExtraPack_t', 
                ]
    
    #family = [  'TestPack_t', ]
    
    def __init__(self, bytes_of_values):
        header = np.frombuffer(bytes_of_values, dtype=HKFILE_H6)
        for x in self.family:
            self.buf = []
            self.buf = header[str(x)]
            self.switch(x)

    def switch(self, arg):
        self.case_name = "case_" + str(arg)
        self.case = getattr(self, self.case_name, lambda:"default")
        return self.case()
 
    def case_AC_HKExtraPack_t(self):
        print(f'{print_bar_indent}   [AC_HKExtraPack_t]')
        struct_data = np.frombuffer(self.buf, dtype=AC_HKExtraPack_t)
        
        for value in range(0, len(struct_data["l0_Status"][0])):
            PrintAndCheck(6, 1, f'l0_Status[{value}]', int(struct_data["l0_Status"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["lastAcceptCmd"][0])):
            PrintAndCheck(6, 1, f'lastAcceptCmd[{value}]', int(struct_data["lastAcceptCmd"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["lastRejCmd"][0])):
            PrintAndCheck(6, 1, f'lastRejCmd[{value}]', int(struct_data["lastRejCmd"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["inertia"][0])):
            PrintAndCheck(6, 1, f'inertia[{value}]', int(struct_data["inertia"][0][value]), ValueType.INT)
            
        PrintAndCheck(6, 1, f'badAttTimer', int(struct_data["badAttTimer"]), ValueType.INT)
        PrintAndCheck(6, 1, f'badRateTimer', int(struct_data["badRateTimer"]), ValueType.INT)
        PrintAndCheck(6, 1, f'cssInvalidCount', int(struct_data["cssInvalidCount"]), ValueType.INT)
        PrintAndCheck(6, 1, f'imuInvalidCount', int(struct_data["imuInvalidCount"]), ValueType.INT)
        PrintAndCheck(6, 1, f'imuReinitCount', int(struct_data["imuReinitCount"]), ValueType.INT)
            
        for value in range(0, len(struct_data["residual"][0])):
            PrintAndCheck(6, 1, f'residual[{value}]', int(struct_data["residual"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["bodyRate"][0])):
            PrintAndCheck(6, 1, f'bodyRate[{value}]', int(struct_data["bodyRate"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["gyroBiasEst"][0])):
            PrintAndCheck(6, 1, f'gyroBiasEst[{value}]', int(struct_data["gyroBiasEst"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["cmdQBodyWrt_ECI"][0])):
            PrintAndCheck(6, 1, f'cmdQBodyWrt_ECI[{value}]', int(struct_data["cmdQBodyWrt_ECI"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["commandedSun"][0])):
            PrintAndCheck(6, 1, f'commandedSun[{value}]', int(struct_data["commandedSun"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["dragEst"][0])):
            PrintAndCheck(6, 1, f'dragEst[{value}]', int(struct_data["dragEst"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["motorFaultCount"][0])):
            PrintAndCheck(6, 1, f'motorFaultCount[{value}]', int(struct_data["motorFaultCount"][0][value]), ValueType.INT)
            
        PrintAndCheck(6, 1, f'sttMedianNoiseAllTrkBlks', int(struct_data["sttMedianNoiseAllTrkBlks"]), ValueType.INT)
        PrintAndCheck(6, 1, f'median_bckGnd', int(struct_data["median_bckGnd"]), ValueType.INT)
        PrintAndCheck(6, 1, f'detTimeoutCount', int(struct_data["detTimeoutCount"]), ValueType.INT)
        PrintAndCheck(6, 1, f'numAttitudeStars', int(struct_data["numAttitudeStars"]), ValueType.INT)
        PrintAndCheck(6, 1, f'eigenError', int(struct_data["eigenError"]), ValueType.INT)
        
        for value in range(0, len(struct_data["sunVectorBody"][0])):
            PrintAndCheck(6, 1, f'sunVectorBody[{value}]', int(struct_data["sunVectorBody"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["magVectorBody"][0])):
            PrintAndCheck(6, 1, f'magVectorBody[{value}]', int(struct_data["magVectorBody"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["rawSunSensorData"][0])):
            PrintAndCheck(6, 1, f'rawSunSensorData[{value}]', int(struct_data["rawSunSensorData"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["rawMagnetometerData"][0])):
            PrintAndCheck(6, 1, f'rawMagnetometerData[{value}]', int(struct_data["rawMagnetometerData"][0][value]), ValueType.INT)
            
        for value in range(0, len(struct_data["imuAvgVector"][0])):
            PrintAndCheck(6, 1, f'imuAvgVector[{value}]', float(struct_data["imuAvgVector"][0][value]), ValueType.FLOAT)
        
        PrintAndCheck(6, 1, f'imuAvgVectorFrame', int(struct_data["imuAvgVectorFrame"]), ValueType.INT)
        PrintAndCheck(6, 1, f'hrRunCount', int(struct_data["hrRunCount"]), ValueType.INT)
        PrintAndCheck(6, 1, f'hrTimeUsec', int(struct_data["hrTimeUsec"]), ValueType.INT)
        PrintAndCheck(6, 1, f'detTemp', int(struct_data["detTemp"]), ValueType.INT)
        PrintAndCheck(6, 1, f'imuTemp', int(struct_data["imuTemp"]), ValueType.INT)
            
        for value in range(0, len(struct_data["motorTemp"][0])):
            PrintAndCheck(6, 1, f'motorTemp[{value}]', int(struct_data["motorTemp"][0][value]), ValueType.INT)
            
        PrintAndCheck(6, 1, f'digitalBus_V', int(struct_data["digitalBus_V"]), ValueType.INT)
        PrintAndCheck(6, 1, f'motorBus_V', int(struct_data["motorBus_V"]), ValueType.INT)
        PrintAndCheck(6, 1, f'rodBus_v', int(struct_data["rodBus_v"]), ValueType.INT)
        PrintAndCheck(6, 1, f'gpsCyclesSinceCRCData', int(struct_data["gpsCyclesSinceCRCData"]), ValueType.INT)
        PrintAndCheck(6, 1, f'gpsCyclesSinceLatestData', int(struct_data["gpsCyclesSinceLatestData"]), ValueType.INT)
        PrintAndCheck(6, 1, f'gpsLockCount', int(struct_data["gpsLockCount"]), ValueType.INT)
        PrintAndCheck(6, 1, f'avgTimeTag', int(struct_data["avgTimeTag"]), ValueType.INT)
        
def GetFileType(filename):
    filetype_key = filename[0:2]
    file_type = 0
    
    if filetype_key == 'H1':
        print(f'{print_indent} HK Type H1')
        file_type = 1
    elif filetype_key == 'H2':
        print(f'{print_indent} HK Type H2')
        file_type = 2
    elif filetype_key == 'H3':
        print(f'{print_indent} HK Type H3')
        file_type = 3
    elif filetype_key == 'H4':
        print(f'{print_indent} HK Type H4')
        file_type = 4
    elif filetype_key == 'H5':
        print(f'{print_indent} HK Type H5')
        file_type = 5
    elif filetype_key == 'H6':
        print(f'{print_indent} HK Type H6')
        file_type = 6
    
    return file_type

def GetStructSize(file_type):
    struct_size = 0
    
    if file_type == 1:
        struct_size += np.dtype(EPS_FSW_HKPack_t).itemsize
        print(f'{print_indent} np.dtype(EPS_FSW_HKPack_t).itemsize : {np.dtype(EPS_FSW_HKPack_t).itemsize}')
        struct_size += np.dtype(CDHS_HKPack_t).itemsize
        print(f'{print_indent} np.dtype(CDHS_HKPack_t).itemsize : {np.dtype(CDHS_HKPack_t).itemsize}')
        struct_size += np.dtype(EPS_SP_HKPack_t).itemsize
        print(f'{print_indent} np.dtype(EPS_SP_HKPack_t).itemsize : {np.dtype(EPS_SP_HKPack_t).itemsize}')
        struct_size += np.dtype(EPS_P60_HKPack_t).itemsize
        print(f'{print_indent} np.dtype(EPS_P60_HKPack_t).itemsize : {np.dtype(EPS_P60_HKPack_t).itemsize}')
        struct_size += np.dtype(CS_AX2150_HKPack_t).itemsize
        print(f'{print_indent} np.dtype(CS_AX2150_HKPack_t).itemsize : {np.dtype(CS_AX2150_HKPack_t).itemsize}')
        
    elif file_type == 2:
        struct_size += np.dtype(AC_HKPack_t).itemsize
        print(f'{print_indent} np.dtype(AC_HKPack_t).itemsize : {np.dtype(AC_HKPack_t).itemsize}')
        
    elif file_type == 3:
        struct_size += np.dtype(CS_EWC27_HKPack_t).itemsize
        print(f'{print_indent} np.dtype(CS_EWC27_HKPack_t).itemsize : {np.dtype(CS_EWC27_HKPack_t).itemsize}')
        
    elif file_type == 4:
        struct_size += np.dtype(PC_PDHS_HKPack_t).itemsize
        print(f'{print_indent} np.dtype(PC_PDHS_HKPack_t).itemsize : {np.dtype(PC_PDHS_HKPack_t).itemsize}')
        
    elif file_type == 5:
        struct_size += np.dtype(PC_PolCube_SOH_t).itemsize
        print(f'{print_indent} np.dtype(PC_PolCube_SOH_t).itemsize : {np.dtype(PC_PolCube_SOH_t).itemsize}')
        struct_size += np.dtype(PC_PolCube_HK_t).itemsize
        print(f'{print_indent} np.dtype(PC_PolCube_HK_t).itemsize : {np.dtype(PC_PolCube_HK_t).itemsize}')
        
    elif file_type == 6:
        struct_size += np.dtype(AC_HKExtraPack_t).itemsize
        print(f'{print_indent} np.dtype(AC_HKExtraPack_t).itemsize : {np.dtype(AC_HKExtraPack_t).itemsize}')
        
    else:
        struct_size = -1
        
    return struct_size

def CheckDummyData(file_type, struct_type, label, value):
    try:
        temp = dummyDB[f'{file_type}.{struct_type}.{label}']
        temp = temp.__str__()
        
        rtn = '\033[31m' + f'------ X ≠ {temp}' + '\033[0m'
        #print(f' --- CheckDummyData check %s' % temp)
        
        if value == temp:
            rtn = ''
        # else:
        #     print(f' --- CheckDummyData unmatch! {value} {temp}')
    except KeyError:
        rtn = '\033[38;2;215;95;215m' + f'------ E : Key Not Found' + '\033[0m'

    return rtn

def CheckDummyFloat(file_type, struct_type, label, value):
    temp = dummyDB[f'{file_type}.{struct_type}.{label}']
    rtn = f'------ X ≠ f {temp.__str__()}'
    #print(f' --- CheckDummyData check %s' % temp)
    
    if math.fabs(value - float(temp)) <= sys.float_info.epsilon:
        rtn = ''
    # else:
    #     print(f' --- CheckDummyData unmatch! {value} {float(temp)}')
        
    return rtn

def PrintAndCheck(file_type, struct_type, label_str, value, value_type, bit_disp_num = 0):
    if print_only_unmatched:
        if value_type == ValueType.HEX:
            chk_str = CheckDummyData(file_type, struct_type, label_str, (value.__str__()))
            if chk_str != '':
                print(' %-35s : %-15s %s ' % (label_str, "0x{0:0{1}x}".format(value, bit_disp_num), chk_str))
        elif value_type == ValueType.FLOAT:
            #value_float = (floor(value*1000)/1000)
            value_float = round(value, 3)
            chk_str = CheckDummyFloat(file_type, struct_type, label_str, value_float)
            if chk_str != '':
                print(' %-35s : %-15s(%-15s) %s ' % (label_str, format(float(value_float), '.3f'), format(float(value)), chk_str))
        else:
            #value_type == ValueType.INT:
            chk_str = CheckDummyData(file_type, struct_type, label_str, (value.__str__()))
            if chk_str != '':
                print(' %-35s : %-15s %s ' % (label_str, int(value), chk_str))
        
    else:
        if value_type == ValueType.HEX:
            chk_str = CheckDummyData(file_type, struct_type, label_str, (value.__str__()))
            print(' %-35s : %-15s %s ' % (label_str, "0x{0:0{1}x}".format(value, bit_disp_num), chk_str))
        elif value_type == ValueType.FLOAT:
            #value_float = (floor(value*1000)/1000)
            value_float = round(value, 3)
            chk_str = CheckDummyFloat(file_type, struct_type, label_str, value_float)
            print(' %-35s : %-15s(%-15s) %s ' % (label_str, format(float(value_float), '.3f'), format(float(value)), chk_str))
        else:
            #value_type == ValueType.INT:
            chk_str = CheckDummyData(file_type, struct_type, label_str, (value.__str__()))
            print(' %-35s : %-15s %s ' % (label_str, int(value), chk_str))

def ReadFile(filepath):
    hex_list = []
    with open(filepath, 'rb') as fp:
        hex_list = [c for c in fp.read()]
        
    return hex_list

def CheckHKFileHeader(file_hex_data):
    family = [ 'CFE_FS_Header_t', 
                'DS_FileHeader_t'
                ]
    buf = []
    header = np.frombuffer(file_hex_data, dtype=HKFILE_HEADER)
    struct_data = np.frombuffer(header["CFE_FS_Header_t"], dtype=CFE_FS_Header_t)
    #print(f'{print_bar_indent}   [CFE_FS_Header_t]')
    #print(f'{print_indent} ContentType       : {struct_data["ContentType"]}')
    content_type = int(struct_data["ContentType"])
    content_type_buf = ((content_type & 0x000000FF) << (8 * 3))
    content_type_buf |= ((content_type & 0x0000FF00) << (8 * 1))
    content_type_buf |= ((content_type & 0x00FF0000) >> (8 * 1))
    content_type_buf |= ((content_type & 0xFF000000) >> (8 * 3))
    
    content_type_ref = int(0x63464531)
    
    if content_type_buf == content_type_ref:
        print(f'{print_indent} content_type match!')
        return True
    else:
        print(f'{print_indent} content_type unmatch! {content_type_buf} ≠ {content_type_ref}')
        return False
    
def PrintFileData(file_type, file_hex_data, file_header_size, struct_size):
    
    start_index = file_header_size
    end_index = start_index + struct_size

    packet_line_count = (len(file_hex_data) - file_header_size)/struct_size
    packet_line_count = floor(packet_line_count)
    print(f'{print_indent} packet_line_count : {packet_line_count}')
    print(f'{print_indent} struct_size       : {struct_size}')
    print('')
    a = input('\033[38;5;208m' + f'Start Packet Show! (press any key)' + '\033[0m')
    
    for i in range(0, packet_line_count):
        file_input_buf = file_hex_data[start_index:end_index]
        
        bytes_of_values = bytes(file_input_buf)
        
        if file_type == 1:
            HKFILE_H1_c(bytes_of_values)
        elif file_type == 2:
            HKFILE_H2_c(bytes_of_values)
        elif file_type == 3:
            HKFILE_H3_c(bytes_of_values)
        elif file_type == 4:
            HKFILE_H4_c(bytes_of_values)
        elif file_type == 5:
            HKFILE_H5_c(bytes_of_values)
        elif file_type == 6:
            HKFILE_H6_c(bytes_of_values)
        else:
            print(f'file type error')
            break
        
        start_index += struct_size
        end_index += struct_size
        print(f'end [{i + 1} / {packet_line_count}]')
        print(f'')
        a = input('\033[38;5;208m' + f'Show Next Packet (any key)? or Exit (q)?' + '\033[0m')
        if a == 'q':
            break
        print(f'')
    
    print(f'{print_bar_indent} ')

def main():
    try:
        global print_only_unmatched
        
        parser = argparse.ArgumentParser(description='Observer HK File Data Extract Application CLI Mode Argument Help...')
        #parser.add_argument('-f', '--filepath', help='input file path', required=True)
        parser.add_argument('-d', '--dirpath', help='input directory path', required=True)
        parser.add_argument('-o', '--only_unmatched', help='print line only unmatched case', action='store_true')
        args = parser.parse_args()
        print('')
        
        if args.only_unmatched:
            print_only_unmatched = True
            print(f'{print_indent} print only unmatched line = {print_only_unmatched}')
        else:
            print_only_unmatched = False
            print(f'{print_indent} print all line = {print_only_unmatched}')
            
        file_full_path_ok = False
        
        ## 디렉토리 경로에서 파일명 선택
        if args.dirpath:
            if os.path.exists(args.dirpath):
                print(f'{print_indent} directory exist')
                
                file_list = os.listdir(args.dirpath)
                file_list.sort()
                file_index = 1
                
                print(f' ______________________________')
                print(f' | index | file name   ')
                print(f' ==============================')
                for file_name in file_list:
                    print(f' |  %03d  | {file_name}' % (file_index))
                    file_index += 1
                
                print(f' ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')
                print('')
                select_num = -1
                for i in range(0, 3):
                    a = input('\033[38;5;208m' + f'Input File Index...' + '\033[0m')
                    if a == 'q':
                        return
                    elif a.isdigit():
                        select_num = int(a)
                        break
                    else:
                        print(f'{print_indent} Input Only Number...')

                if select_num == -1:
                    print(f'{print_indent} 3 strike out.. Bye~')
                    return
                else:
                    file_full_path = f'{args.dirpath}/{file_list[select_num - 1]}'
                    file_full_path_ok = True
                    #print(f'{print_indent} file_full_path : {file_full_path}')
            else:
                print(f'{print_indent} directory not exist')
        else:
            print(f'{print_indent} directory path argument is empty!')
        
        ## 파일명 바탕으로 추출
        if file_full_path_ok:
            print('')
            print(f'{print_indent} input filepath : {file_full_path}')
            rtn = os.path.isfile(file_full_path)
            
            if rtn:
                print(f'{print_indent} file exist check')
                file_hex = ReadFile(file_full_path)
                file_hex_size = len(file_hex)
                print(f'{print_indent} file length : {file_hex_size}')
                
                #print(f'{print_indent} file_header_size1 : {np.dtype(CFE_FS_Header_t).itemsize}')
                #print(f'{print_indent} file_header_size2 : {np.dtype(DS_FileHeader_t).itemsize}')
                
                file_header_input_buf = file_hex[0:76]
                file_header_input_buf_bytes = bytes(file_header_input_buf)
        
                is_has_header = CheckHKFileHeader(file_header_input_buf_bytes)
                global file_header_size 
                if is_has_header:
                    file_header_size = 140
                else:
                    file_header_size = 0
                    
                expect_file_size = file_header_size
                struct_size = 0
                
                file_type = GetFileType(os.path.basename(file_full_path))
                
                struct_size = GetStructSize(file_type)
                
                if struct_size < 1:
                    print(f' unsupported file type')
                    return
                
                expect_file_size = file_header_size + struct_size
                
                if file_hex_size == 0:
                    print(f'{print_indent} file empty')
                elif file_hex_size < expect_file_size:
                    print(f'{print_indent} file size is less than size expected {file_hex_size} < {expect_file_size}')
                else:
                    print(f'{print_indent} file size OK')
                    print('')
                    PrintFileData(file_type, file_hex, file_header_size, struct_size)
                
            else:
                print(f'{print_indent} file not exist...')
    
    except Exception as ex: 
        print(f'exp : {ex}')

if __name__ == "__main__":
    
    ##############################################################################
    dummyDB["1.1.selectedSband"] = 0
    dummyDB["1.1.sbandStatus_P"] = 1
    dummyDB["1.1.sbandStatus_R"] = 0
    dummyDB["1.1.spare"] = 0x13
    dummyDB["1.1.operationMode"] = 2
    dummyDB["1.1.operationSubMode"] = 3
    dummyDB["1.1.rebootCount"] = 4
    dummyDB["1.1.nandFlashCapacity"] = 5
    dummyDB["1.1.sdCardCapacity"] = 6
    
    dummyDB["1.2.temperature"] = -20000
    dummyDB["1.2.edacCount"] = 20001
    
    dummyDB["1.3.depStatus"] = 0x33
    dummyDB["1.3.depStatus_GPIO"] = 0x33
    dummyDB["1.3.spare"] = 0x09
    dummyDB["1.3.panelTemp_ST_0"] = -10101
    dummyDB["1.3.panelTemp_ST_1"] = -10102
    dummyDB["1.3.panelTemp_ST_2"] = -10103
    dummyDB["1.3.panelTemp_ST_3"] = -10104
    dummyDB["1.3.panelTemp_UTT_0"] = -10105
    dummyDB["1.3.panelTemp_UTT_1"] = -10106
    dummyDB["1.3.panelTemp_UTT_2"] = -10107
    dummyDB["1.3.panelTemp_UBT_0"] = -10108
    dummyDB["1.3.panelTemp_UBT_1"] = -10109
    dummyDB["1.3.panelTemp_UBT_2"] = -10110
    dummyDB["1.3.antennaTemp_0"] = -10111
    dummyDB["1.3.antennaTemp_1"] = -10112
    dummyDB["1.3.eocTemp_0"] = -10113
    dummyDB["1.3.eocTemp_1"] = -10114
    
    dummyDB["1.4.dockDeviceStatus"] = 0x801
    dummyDB["1.4.acuDeviceStatus"] = 0x3801
    dummyDB["1.4.heaterStatus"] = 0x3
    dummyDB["1.4.spare1"] = 0x0
    dummyDB["1.4.pduDeviceStatus"] = 0x81
    
    dummyDB["1.4.powerSwitchStatus"] = 0x49
    dummyDB["1.4.xbandSwitchStatus"] = 0x3
    dummyDB["1.4.pdhsSwitchStatus"] = 0x0
    dummyDB["1.4.polcubeSwitchStatus"] = 0x3
    dummyDB["1.4.spare6"] = 0x5
    
    dummyDB["1.4.antXPHeaterSwitchStatus"] = 0x0
    dummyDB["1.4.antXMHeaterSwitchStatus"] = 0x3
    dummyDB["1.4.eocHeaterSwitchStatus"] = 0x0
    dummyDB["1.4.spare7"] = 0x0
    
    dummyDB["1.4.SwitchCurrent[0]"] = -10001
    dummyDB["1.4.SwitchCurrent[1]"] = -10002
    dummyDB["1.4.SwitchCurrent[2]"] = -10003
    dummyDB["1.4.SwitchCurrent[3]"] = -10004
    dummyDB["1.4.SwitchCurrent[4]"] = -10005
    dummyDB["1.4.SwitchCurrent[5]"] = -10006
    dummyDB["1.4.SwitchCurrent[6]"] = -10007
    dummyDB["1.4.SwitchVoltage[0]"] = 10008
    dummyDB["1.4.SwitchVoltage[1]"] = 10009
    dummyDB["1.4.SwitchVoltage[2]"] = 10010
    dummyDB["1.4.SwitchVoltage[3]"] = 10011
    dummyDB["1.4.SwitchVoltage[4]"] = 10012
    dummyDB["1.4.SwitchVoltage[5]"] = 10013
    dummyDB["1.4.SwitchVoltage[6]"] = 10014
    dummyDB["1.4.SwitchlatchupCount[0]"] = 10015
    dummyDB["1.4.SwitchlatchupCount[1]"] = 10016
    dummyDB["1.4.SwitchlatchupCount[2]"] = 10017
    dummyDB["1.4.SwitchlatchupCount[3]"] = 10018
    dummyDB["1.4.SwitchlatchupCount[4]"] = 10019
    dummyDB["1.4.SwitchlatchupCount[5]"] = 10020
    dummyDB["1.4.SwitchlatchupCount[6]"] = 10021
    dummyDB["1.4.docklatchupCount[0]"] = 10022
    dummyDB["1.4.docklatchupCount[1]"] = 10023
    dummyDB["1.4.docklatchupCount[2]"] = 10024
    dummyDB["1.4.docklatchupCount[3]"] = 10025
    dummyDB["1.4.acuCurrent[0]"] = -10026
    dummyDB["1.4.acuCurrent[1]"] = -10027
    dummyDB["1.4.acuCurrent[2]"] = -10028
    dummyDB["1.4.acuCurrent[3]"] = -10029
    dummyDB["1.4.acuCurrent[4]"] = -10030
    dummyDB["1.4.acuCurrent[5]"] = -10031
    dummyDB["1.4.acuVoltage[0]"] = 10032
    dummyDB["1.4.acuVoltage[1]"] = 10033
    dummyDB["1.4.acuVoltage[2]"] = 10034
    dummyDB["1.4.acuVoltage[3]"] = 10035
    dummyDB["1.4.acuVoltage[4]"] = 10036
    dummyDB["1.4.acuVoltage[5]"] = 10037
    dummyDB["1.4.dockTemp[0]"] = -10038
    dummyDB["1.4.dockTemp[1]"] = -10039
    dummyDB["1.4.pduTemp"] = -10040
    dummyDB["1.4.acuTemp[0]"] = -10041
    dummyDB["1.4.acuTemp[1]"] = -10042
    dummyDB["1.4.acuTemp[2]"] = -10043
    dummyDB["1.4.bat_temperature[0]"] = -10044
    dummyDB["1.4.bat_temperature[1]"] = -10045
    dummyDB["1.4.bat_temperature[2]"] = -10046
    dummyDB["1.4.bat_temperature[3]"] = -10047
    dummyDB["1.4.bat_temperature[4]"] = -10048
    dummyDB["1.4.bat_temperature[5]"] = -10049
    dummyDB["1.4.bat_temperature[6]"] = -10050
    dummyDB["1.4.bat_temperature[7]"] = -10051
    dummyDB["1.4.vbatt"] = 10052
    dummyDB["1.4.chargeCurrent"] = 10053
    dummyDB["1.4.dischargeCurrent"] = 10054
    dummyDB["1.4.dockBootcause"] = 55
    dummyDB["1.4.acuBootcause"] = 56
    dummyDB["1.4.pduBootcause"] = 57
    
    dummyDB["1.5.bootCause"] = 0x9
    dummyDB["1.5.spare"] = 0x9
    dummyDB["1.5.temperature_board"] = -3521
    dummyDB["1.5.temperature_pa"] = 0
    dummyDB["1.5.totalTxBytes"] = 0xff0120ff
    dummyDB["1.5.totalRxBytes"] = 0xff0230ff
    dummyDB["1.5.bootCount"] = 0xf45f
    
    ##############################################################################
    
    dummyDB["2.1.l0_Status"] = 0x41
    dummyDB["2.1.momentumHealth"] = 0x11 
    dummyDB["2.1.magSourceUsed"] = 0x9
    dummyDB["2.1.timeValid"] = 0x1
    dummyDB["2.1.refs_valid"] = 0x0
    dummyDB["2.1.attCtrlHealth"] = 0x3
    dummyDB["2.1.sttOperatingMode"] = 0x0
    dummyDB["2.1.torqueRodeMode1"] = 0x9
    dummyDB["2.1.torqueRodeMode2"] = 0x0
    dummyDB["2.1.torqueRodeMode3"] = 0x9
    dummyDB["2.1.torqueRodeFiringPack"] = 0x00
    dummyDB["2.1.tableUploadStatus"] = 0x11
    dummyDB["2.1.spare1"] = 0x0
    dummyDB["2.1.sunPoint_state"] = 0x5
    dummyDB["2.1.attDetHealth"] = 0x0
    dummyDB["2.1.attCmdHealth"] = 0x11
    dummyDB["2.1.attStatus"] = 0x00
    dummyDB["2.1.runLowRateTask"] = 0x1
    dummyDB["2.1.magHealth"] = 0x0
    dummyDB["2.1.cssHealth"] = 0x3
    dummyDB["2.1.imuHealth"] = 0x0
    dummyDB["2.1.gpsHealth"] = 0x21
    dummyDB["2.1.spare2"] = 0x0
    dummyDB["2.1.taiSeconds"] = 200001
    dummyDB["2.1.positionWrt_ECI[0]"] = 200002.001
    dummyDB["2.1.positionWrt_ECI[1]"] = 200003.002
    dummyDB["2.1.positionWrt_ECI[2]"] = 200004.003
    dummyDB["2.1.velocityWrt_ECI[0]"] = 200005.004
    dummyDB["2.1.velocityWrt_ECI[1]"] = 200006.005
    dummyDB["2.1.velocityWrt_ECI[2]"] = 200007.006
    dummyDB["2.1.qBodyWrt_ECI[0]"] = 200008
    dummyDB["2.1.qBodyWrt_ECI[1]"] = 200009
    dummyDB["2.1.qBodyWrt_ECI[2]"] = 200010
    dummyDB["2.1.qBodyWrt_ECI[3]"] = 200011
    dummyDB["2.1.filteredSpeed_RPM[0]"] = 20012
    dummyDB["2.1.filteredSpeed_RPM[1]"] = 20013
    dummyDB["2.1.filteredSpeed_RPM[2]"] = 20014
    dummyDB["2.1.totalMomentumMag"] = 20015.007
    
    ##############################################################################
    
    dummyDB["3.1.sourceOfLastStartup"] = 0x201
    dummyDB["3.1.stateOfUnit"] = 0x0
    dummyDB["3.1.selfTestResult"] = 0x3
    dummyDB["3.1.temperature"] = -31
    
    ##############################################################################
    
    dummyDB["4.1.errorStatus"] = 41
    dummyDB["4.1.temperature"] = -25535
    dummyDB["4.1.receivedCommandCount"] = 400002
    dummyDB["4.1.receivedCommandErrorCount"] = 400003
    
    ##############################################################################
    
    dummyDB["5.1.cameraId"] = 0x1
    dummyDB["5.1.binningStatus"] = 0x0
    dummyDB["5.1.ldoStatus"] = 0x1
    dummyDB["5.1.sensorSyncStatus"] = 0x0
    dummyDB["5.1.reserved"] = 0x009
    
    dummyDB["5.2.currentTime"] = 50002
    dummyDB["5.2.ldoBoardTemp"] = 50003
    dummyDB["5.2.powerBoardTemp"] = 50004
    dummyDB["5.2.ScifBoardTemp"] = 50005
    dummyDB["5.2.FpgaTemp"] = 50006
    dummyDB["5.2.SensorTemp"] = 50007
    
    ##############################################################################
    
    dummyDB["6.1.l0_Status[0]"] = 200017
    dummyDB["6.1.l0_Status[1]"] = 200018
    dummyDB["6.1.l0_Status[2]"] = 200019
    dummyDB["6.1.l0_Status[3]"] = 200020
    dummyDB["6.1.l0_Status[4]"] = 200021
    dummyDB["6.1.l0_Status[5]"] = 200022
    dummyDB["6.1.l0_Status[6]"] = 200023
    dummyDB["6.1.lastAcceptCmd[0]"] = 224
    dummyDB["6.1.lastAcceptCmd[1]"] = 225
    dummyDB["6.1.lastAcceptCmd[2]"] = 226
    dummyDB["6.1.lastRejCmd[0]"] = 227
    dummyDB["6.1.lastRejCmd[1]"] = 228
    dummyDB["6.1.lastRejCmd[2]"] = 229
    dummyDB["6.1.inertia[0]"] = 200030
    dummyDB["6.1.inertia[1]"] = 200031
    dummyDB["6.1.inertia[2]"] = 200032
    dummyDB["6.1.badAttTimer"] = 200033
    dummyDB["6.1.badRateTimer"] = 200034
    dummyDB["6.1.cssInvalidCount"] = 20035
    dummyDB["6.1.imuInvalidCount"] = 20036
    dummyDB["6.1.imuReinitCount"] = 20037
    dummyDB["6.1.residual[0]"] = 200038
    dummyDB["6.1.residual[1]"] = 200039
    dummyDB["6.1.residual[2]"] = 200040
    dummyDB["6.1.bodyRate[0]"] = 200041
    dummyDB["6.1.bodyRate[1]"] = 200042
    dummyDB["6.1.bodyRate[2]"] = 200043
    dummyDB["6.1.gyroBiasEst[0]"] = 20044
    dummyDB["6.1.gyroBiasEst[1]"] = 20045
    dummyDB["6.1.gyroBiasEst[2]"] = 20046
    dummyDB["6.1.cmdQBodyWrt_ECI[0]"] = 200047
    dummyDB["6.1.cmdQBodyWrt_ECI[1]"] = 200048
    dummyDB["6.1.cmdQBodyWrt_ECI[2]"] = 200049
    dummyDB["6.1.cmdQBodyWrt_ECI[3]"] = 200050
    dummyDB["6.1.commandedSun[0]"] = 20051
    dummyDB["6.1.commandedSun[1]"] = 20052
    dummyDB["6.1.commandedSun[2]"] = 20053
    dummyDB["6.1.dragEst[0]"] = 20054
    dummyDB["6.1.dragEst[1]"] = 20055
    dummyDB["6.1.dragEst[2]"] = 20056
    dummyDB["6.1.motorFaultCount[0]"] = 57
    dummyDB["6.1.motorFaultCount[1]"] = 58
    dummyDB["6.1.motorFaultCount[2]"] = 59
    dummyDB["6.1.sttMedianNoiseAllTrkBlks"] = 60
    dummyDB["6.1.median_bckGnd"] = 51
    dummyDB["6.1.detTimeoutCount"] = 20061
    dummyDB["6.1.numAttitudeStars"] = 62
    dummyDB["6.1.eigenError"] = 200063
    dummyDB["6.1.sunVectorBody[0]"] = 20064
    dummyDB["6.1.sunVectorBody[1]"] = 20065
    dummyDB["6.1.sunVectorBody[2]"] = 20066
    dummyDB["6.1.magVectorBody[0]"] = 20067
    dummyDB["6.1.magVectorBody[1]"] = 20068
    dummyDB["6.1.magVectorBody[2]"] = 20069
    
    dummyDB["6.1.rawSunSensorData[0]"] = 20070
    dummyDB["6.1.rawSunSensorData[1]"] = 20071
    dummyDB["6.1.rawSunSensorData[2]"] = 20072
    dummyDB["6.1.rawSunSensorData[3]"] = 20073
    dummyDB["6.1.rawSunSensorData[4]"] = 20074
    dummyDB["6.1.rawSunSensorData[5]"] = 20075
    dummyDB["6.1.rawSunSensorData[6]"] = 20076
    dummyDB["6.1.rawSunSensorData[7]"] = 20077
    dummyDB["6.1.rawSunSensorData[8]"] = 20078
    dummyDB["6.1.rawSunSensorData[9]"] = 20079
    dummyDB["6.1.rawSunSensorData[10]"] = 20080
    dummyDB["6.1.rawSunSensorData[11]"] = 20081
    dummyDB["6.1.rawMagnetometerData[0]"] = 20082
    dummyDB["6.1.rawMagnetometerData[1]"] = 20083
    dummyDB["6.1.rawMagnetometerData[2]"] = 20084
    dummyDB["6.1.rawMagnetometerData[3]"] = 20085
    dummyDB["6.1.rawMagnetometerData[4]"] = 20086
    dummyDB["6.1.rawMagnetometerData[5]"] = 20087
    dummyDB["6.1.rawMagnetometerData[6]"] = 20088
    dummyDB["6.1.rawMagnetometerData[7]"] = 20089
    dummyDB["6.1.rawMagnetometerData[8]"] = 20090
    dummyDB["6.1.imuAvgVector[0]"] = 20091.008
    dummyDB["6.1.imuAvgVector[1]"] = 20092.009
    dummyDB["6.1.imuAvgVector[2]"] = 20093.010
    dummyDB["6.1.imuAvgVectorFrame"] = 94
    dummyDB["6.1.hrRunCount"] = 200095
    dummyDB["6.1.hrTimeUsec"] = 200096
    dummyDB["6.1.detTemp"] = 97
    dummyDB["6.1.imuTemp"] = 20098
    dummyDB["6.1.motorTemp[0]"] = 20099
    dummyDB["6.1.motorTemp[1]"] = 20100
    dummyDB["6.1.motorTemp[2]"] = 20101
    dummyDB["6.1.digitalBus_V"] = 20102
    dummyDB["6.1.motorBus_V"] = 20103
    dummyDB["6.1.rodBus_v"] = 20104
    dummyDB["6.1.gpsCyclesSinceCRCData"] = 200105
    dummyDB["6.1.gpsCyclesSinceLatestData"] = 200106
    dummyDB["6.1.gpsLockCount"] = 20107
    dummyDB["6.1.avgTimeTag"] = 200108
    
    print("\033[38;5;208mCheck Observer HK File Data !\033[0m")
    main()
    