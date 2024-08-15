// RUN: triton-tensor-layout -i %s -alias-names="blocked" -t "tensor<16x16xf16>" | FileCheck %s --check-prefix=CHECK-BLOCKED

// RUN: triton-tensor-layout -i %s -alias-names="mfma" -t "tensor<16x16xf16>" | FileCheck %s --check-prefix=CHECK-MFMA

// RUN: triton-tensor-layout -l "#triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>" -t "tensor<16x16xf16>" | FileCheck %s --check-prefix=CHECK-MFMA

// RUN: triton-tensor-layout -i %s -alias-names="mfma" -t "tensor<16x16xf16>" -use-hw-view | FileCheck %s --check-prefix=CHECK-HW

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
tt.func @print(%A : !tt.ptr<f16>) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #blocked>
  %cst1 = arith.constant dense<0.00e+00> : tensor<16x16xf16, #mfma>
  tt.return
}

// CHECK-BLOCKED: Print layout attribute: #blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-BLOCKED: T0:0|  T4:0,   T0:1|  T4:1,   T0:2|  T4:2,   T0:3|  T4:3,   T1:0|  T5:0,   T1:1|  T5:1,   T1:2|  T5:2,   T1:3|  T5:3,   T2:0|  T6:0,   T2:1|  T6:1,   T2:2|  T6:2,   T2:3|  T6:3,   T3:0|  T7:0,   T3:1|  T7:1,   T3:2|  T7:2,   T3:3|  T7:3
// CHECK-BLOCKED: T8:0| T12:0,   T8:1| T12:1,   T8:2| T12:2,   T8:3| T12:3,   T9:0| T13:0,   T9:1| T13:1,   T9:2| T13:2,   T9:3| T13:3,  T10:0| T14:0,  T10:1| T14:1,  T10:2| T14:2,  T10:3| T14:3,  T11:0| T15:0,  T11:1| T15:1,  T11:2| T15:2,  T11:3| T15:3
// CHECK-BLOCKED: T16:0| T20:0,  T16:1| T20:1,  T16:2| T20:2,  T16:3| T20:3,  T17:0| T21:0,  T17:1| T21:1,  T17:2| T21:2,  T17:3| T21:3,  T18:0| T22:0,  T18:1| T22:1,  T18:2| T22:2,  T18:3| T22:3,  T19:0| T23:0,  T19:1| T23:1,  T19:2| T23:2,  T19:3| T23:3
// CHECK-BLOCKED: T24:0| T28:0,  T24:1| T28:1,  T24:2| T28:2,  T24:3| T28:3,  T25:0| T29:0,  T25:1| T29:1,  T25:2| T29:2,  T25:3| T29:3,  T26:0| T30:0,  T26:1| T30:1,  T26:2| T30:2,  T26:3| T30:3,  T27:0| T31:0,  T27:1| T31:1,  T27:2| T31:2,  T27:3| T31:3
// CHECK-BLOCKED: T32:0| T36:0,  T32:1| T36:1,  T32:2| T36:2,  T32:3| T36:3,  T33:0| T37:0,  T33:1| T37:1,  T33:2| T37:2,  T33:3| T37:3,  T34:0| T38:0,  T34:1| T38:1,  T34:2| T38:2,  T34:3| T38:3,  T35:0| T39:0,  T35:1| T39:1,  T35:2| T39:2,  T35:3| T39:3
// CHECK-BLOCKED: T40:0| T44:0,  T40:1| T44:1,  T40:2| T44:2,  T40:3| T44:3,  T41:0| T45:0,  T41:1| T45:1,  T41:2| T45:2,  T41:3| T45:3,  T42:0| T46:0,  T42:1| T46:1,  T42:2| T46:2,  T42:3| T46:3,  T43:0| T47:0,  T43:1| T47:1,  T43:2| T47:2,  T43:3| T47:3
// CHECK-BLOCKED: T48:0| T52:0,  T48:1| T52:1,  T48:2| T52:2,  T48:3| T52:3,  T49:0| T53:0,  T49:1| T53:1,  T49:2| T53:2,  T49:3| T53:3,  T50:0| T54:0,  T50:1| T54:1,  T50:2| T54:2,  T50:3| T54:3,  T51:0| T55:0,  T51:1| T55:1,  T51:2| T55:2,  T51:3| T55:3
// CHECK-BLOCKED: T56:0| T60:0,  T56:1| T60:1,  T56:2| T60:2,  T56:3| T60:3,  T57:0| T61:0,  T57:1| T61:1,  T57:2| T61:2,  T57:3| T61:3,  T58:0| T62:0,  T58:1| T62:1,  T58:2| T62:2,  T58:3| T62:3,  T59:0| T63:0,  T59:1| T63:1,  T59:2| T63:2,  T59:3| T63:3
// CHECK-BLOCKED: T64:0| T68:0,  T64:1| T68:1,  T64:2| T68:2,  T64:3| T68:3,  T65:0| T69:0,  T65:1| T69:1,  T65:2| T69:2,  T65:3| T69:3,  T66:0| T70:0,  T66:1| T70:1,  T66:2| T70:2,  T66:3| T70:3,  T67:0| T71:0,  T67:1| T71:1,  T67:2| T71:2,  T67:3| T71:3
// CHECK-BLOCKED: T72:0| T76:0,  T72:1| T76:1,  T72:2| T76:2,  T72:3| T76:3,  T73:0| T77:0,  T73:1| T77:1,  T73:2| T77:2,  T73:3| T77:3,  T74:0| T78:0,  T74:1| T78:1,  T74:2| T78:2,  T74:3| T78:3,  T75:0| T79:0,  T75:1| T79:1,  T75:2| T79:2,  T75:3| T79:3
// CHECK-BLOCKED: T80:0| T84:0,  T80:1| T84:1,  T80:2| T84:2,  T80:3| T84:3,  T81:0| T85:0,  T81:1| T85:1,  T81:2| T85:2,  T81:3| T85:3,  T82:0| T86:0,  T82:1| T86:1,  T82:2| T86:2,  T82:3| T86:3,  T83:0| T87:0,  T83:1| T87:1,  T83:2| T87:2,  T83:3| T87:3
// CHECK-BLOCKED: T88:0| T92:0,  T88:1| T92:1,  T88:2| T92:2,  T88:3| T92:3,  T89:0| T93:0,  T89:1| T93:1,  T89:2| T93:2,  T89:3| T93:3,  T90:0| T94:0,  T90:1| T94:1,  T90:2| T94:2,  T90:3| T94:3,  T91:0| T95:0,  T91:1| T95:1,  T91:2| T95:2,  T91:3| T95:3
// CHECK-BLOCKED: T96:0|T100:0,  T96:1|T100:1,  T96:2|T100:2,  T96:3|T100:3,  T97:0|T101:0,  T97:1|T101:1,  T97:2|T101:2,  T97:3|T101:3,  T98:0|T102:0,  T98:1|T102:1,  T98:2|T102:2,  T98:3|T102:3,  T99:0|T103:0,  T99:1|T103:1,  T99:2|T103:2,  T99:3|T103:3
// CHECK-BLOCKED: T104:0|T108:0, T104:1|T108:1, T104:2|T108:2, T104:3|T108:3, T105:0|T109:0, T105:1|T109:1, T105:2|T109:2, T105:3|T109:3, T106:0|T110:0, T106:1|T110:1, T106:2|T110:2, T106:3|T110:3, T107:0|T111:0, T107:1|T111:1, T107:2|T111:2, T107:3|T111:3
// CHECK-BLOCKED: T112:0|T116:0, T112:1|T116:1, T112:2|T116:2, T112:3|T116:3, T113:0|T117:0, T113:1|T117:1, T113:2|T117:2, T113:3|T117:3, T114:0|T118:0, T114:1|T118:1, T114:2|T118:2, T114:3|T118:3, T115:0|T119:0, T115:1|T119:1, T115:2|T119:2, T115:3|T119:3
// CHECK-BLOCKED: T120:0|T124:0, T120:1|T124:1, T120:2|T124:2, T120:3|T124:3, T121:0|T125:0, T121:1|T125:1, T121:2|T125:2, T121:3|T125:3, T122:0|T126:0, T122:1|T126:1, T122:2|T126:2, T122:3|T126:3, T123:0|T127:0, T123:1|T127:1, T123:2|T127:2, T123:3|T127:3


// CHECK-MFMA: Print layout attribute: {{.*}}#triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
// CHECK-MFMA: T0:0| T64:0|T128:0|T192:0,   T0:1| T64:1|T128:1|T192:1,   T0:2| T64:2|T128:2|T192:2,   T0:3| T64:3|T128:3|T192:3,  T16:0| T80:0|T144:0|T208:0,  T16:1| T80:1|T144:1|T208:1,  T16:2| T80:2|T144:2|T208:2,  T16:3| T80:3|T144:3|T208:3,  T32:0| T96:0|T160:0|T224:0,  T32:1| T96:1|T160:1|T224:1,  T32:2| T96:2|T160:2|T224:2,  T32:3| T96:3|T160:3|T224:3,  T48:0|T112:0|T176:0|T240:0,  T48:1|T112:1|T176:1|T240:1,  T48:2|T112:2|T176:2|T240:2,  T48:3|T112:3|T176:3|T240:3
// CHECK-MFMA: T1:0| T65:0|T129:0|T193:0,   T1:1| T65:1|T129:1|T193:1,   T1:2| T65:2|T129:2|T193:2,   T1:3| T65:3|T129:3|T193:3,  T17:0| T81:0|T145:0|T209:0,  T17:1| T81:1|T145:1|T209:1,  T17:2| T81:2|T145:2|T209:2,  T17:3| T81:3|T145:3|T209:3,  T33:0| T97:0|T161:0|T225:0,  T33:1| T97:1|T161:1|T225:1,  T33:2| T97:2|T161:2|T225:2,  T33:3| T97:3|T161:3|T225:3,  T49:0|T113:0|T177:0|T241:0,  T49:1|T113:1|T177:1|T241:1,  T49:2|T113:2|T177:2|T241:2,  T49:3|T113:3|T177:3|T241:3
// CHECK-MFMA: T2:0| T66:0|T130:0|T194:0,   T2:1| T66:1|T130:1|T194:1,   T2:2| T66:2|T130:2|T194:2,   T2:3| T66:3|T130:3|T194:3,  T18:0| T82:0|T146:0|T210:0,  T18:1| T82:1|T146:1|T210:1,  T18:2| T82:2|T146:2|T210:2,  T18:3| T82:3|T146:3|T210:3,  T34:0| T98:0|T162:0|T226:0,  T34:1| T98:1|T162:1|T226:1,  T34:2| T98:2|T162:2|T226:2,  T34:3| T98:3|T162:3|T226:3,  T50:0|T114:0|T178:0|T242:0,  T50:1|T114:1|T178:1|T242:1,  T50:2|T114:2|T178:2|T242:2,  T50:3|T114:3|T178:3|T242:3
// CHECK-MFMA: T3:0| T67:0|T131:0|T195:0,   T3:1| T67:1|T131:1|T195:1,   T3:2| T67:2|T131:2|T195:2,   T3:3| T67:3|T131:3|T195:3,  T19:0| T83:0|T147:0|T211:0,  T19:1| T83:1|T147:1|T211:1,  T19:2| T83:2|T147:2|T211:2,  T19:3| T83:3|T147:3|T211:3,  T35:0| T99:0|T163:0|T227:0,  T35:1| T99:1|T163:1|T227:1,  T35:2| T99:2|T163:2|T227:2,  T35:3| T99:3|T163:3|T227:3,  T51:0|T115:0|T179:0|T243:0,  T51:1|T115:1|T179:1|T243:1,  T51:2|T115:2|T179:2|T243:2,  T51:3|T115:3|T179:3|T243:3
// CHECK-MFMA: T4:0| T68:0|T132:0|T196:0,   T4:1| T68:1|T132:1|T196:1,   T4:2| T68:2|T132:2|T196:2,   T4:3| T68:3|T132:3|T196:3,  T20:0| T84:0|T148:0|T212:0,  T20:1| T84:1|T148:1|T212:1,  T20:2| T84:2|T148:2|T212:2,  T20:3| T84:3|T148:3|T212:3,  T36:0|T100:0|T164:0|T228:0,  T36:1|T100:1|T164:1|T228:1,  T36:2|T100:2|T164:2|T228:2,  T36:3|T100:3|T164:3|T228:3,  T52:0|T116:0|T180:0|T244:0,  T52:1|T116:1|T180:1|T244:1,  T52:2|T116:2|T180:2|T244:2,  T52:3|T116:3|T180:3|T244:3
// CHECK-MFMA: T5:0| T69:0|T133:0|T197:0,   T5:1| T69:1|T133:1|T197:1,   T5:2| T69:2|T133:2|T197:2,   T5:3| T69:3|T133:3|T197:3,  T21:0| T85:0|T149:0|T213:0,  T21:1| T85:1|T149:1|T213:1,  T21:2| T85:2|T149:2|T213:2,  T21:3| T85:3|T149:3|T213:3,  T37:0|T101:0|T165:0|T229:0,  T37:1|T101:1|T165:1|T229:1,  T37:2|T101:2|T165:2|T229:2,  T37:3|T101:3|T165:3|T229:3,  T53:0|T117:0|T181:0|T245:0,  T53:1|T117:1|T181:1|T245:1,  T53:2|T117:2|T181:2|T245:2,  T53:3|T117:3|T181:3|T245:3
// CHECK-MFMA: T6:0| T70:0|T134:0|T198:0,   T6:1| T70:1|T134:1|T198:1,   T6:2| T70:2|T134:2|T198:2,   T6:3| T70:3|T134:3|T198:3,  T22:0| T86:0|T150:0|T214:0,  T22:1| T86:1|T150:1|T214:1,  T22:2| T86:2|T150:2|T214:2,  T22:3| T86:3|T150:3|T214:3,  T38:0|T102:0|T166:0|T230:0,  T38:1|T102:1|T166:1|T230:1,  T38:2|T102:2|T166:2|T230:2,  T38:3|T102:3|T166:3|T230:3,  T54:0|T118:0|T182:0|T246:0,  T54:1|T118:1|T182:1|T246:1,  T54:2|T118:2|T182:2|T246:2,  T54:3|T118:3|T182:3|T246:3
// CHECK-MFMA: T7:0| T71:0|T135:0|T199:0,   T7:1| T71:1|T135:1|T199:1,   T7:2| T71:2|T135:2|T199:2,   T7:3| T71:3|T135:3|T199:3,  T23:0| T87:0|T151:0|T215:0,  T23:1| T87:1|T151:1|T215:1,  T23:2| T87:2|T151:2|T215:2,  T23:3| T87:3|T151:3|T215:3,  T39:0|T103:0|T167:0|T231:0,  T39:1|T103:1|T167:1|T231:1,  T39:2|T103:2|T167:2|T231:2,  T39:3|T103:3|T167:3|T231:3,  T55:0|T119:0|T183:0|T247:0,  T55:1|T119:1|T183:1|T247:1,  T55:2|T119:2|T183:2|T247:2,  T55:3|T119:3|T183:3|T247:3
// CHECK-MFMA: T8:0| T72:0|T136:0|T200:0,   T8:1| T72:1|T136:1|T200:1,   T8:2| T72:2|T136:2|T200:2,   T8:3| T72:3|T136:3|T200:3,  T24:0| T88:0|T152:0|T216:0,  T24:1| T88:1|T152:1|T216:1,  T24:2| T88:2|T152:2|T216:2,  T24:3| T88:3|T152:3|T216:3,  T40:0|T104:0|T168:0|T232:0,  T40:1|T104:1|T168:1|T232:1,  T40:2|T104:2|T168:2|T232:2,  T40:3|T104:3|T168:3|T232:3,  T56:0|T120:0|T184:0|T248:0,  T56:1|T120:1|T184:1|T248:1,  T56:2|T120:2|T184:2|T248:2,  T56:3|T120:3|T184:3|T248:3
// CHECK-MFMA: T9:0| T73:0|T137:0|T201:0,   T9:1| T73:1|T137:1|T201:1,   T9:2| T73:2|T137:2|T201:2,   T9:3| T73:3|T137:3|T201:3,  T25:0| T89:0|T153:0|T217:0,  T25:1| T89:1|T153:1|T217:1,  T25:2| T89:2|T153:2|T217:2,  T25:3| T89:3|T153:3|T217:3,  T41:0|T105:0|T169:0|T233:0,  T41:1|T105:1|T169:1|T233:1,  T41:2|T105:2|T169:2|T233:2,  T41:3|T105:3|T169:3|T233:3,  T57:0|T121:0|T185:0|T249:0,  T57:1|T121:1|T185:1|T249:1,  T57:2|T121:2|T185:2|T249:2,  T57:3|T121:3|T185:3|T249:3
// CHECK-MFMA: T10:0| T74:0|T138:0|T202:0,  T10:1| T74:1|T138:1|T202:1,  T10:2| T74:2|T138:2|T202:2,  T10:3| T74:3|T138:3|T202:3,  T26:0| T90:0|T154:0|T218:0,  T26:1| T90:1|T154:1|T218:1,  T26:2| T90:2|T154:2|T218:2,  T26:3| T90:3|T154:3|T218:3,  T42:0|T106:0|T170:0|T234:0,  T42:1|T106:1|T170:1|T234:1,  T42:2|T106:2|T170:2|T234:2,  T42:3|T106:3|T170:3|T234:3,  T58:0|T122:0|T186:0|T250:0,  T58:1|T122:1|T186:1|T250:1,  T58:2|T122:2|T186:2|T250:2,  T58:3|T122:3|T186:3|T250:3
// CHECK-MFMA: T11:0| T75:0|T139:0|T203:0,  T11:1| T75:1|T139:1|T203:1,  T11:2| T75:2|T139:2|T203:2,  T11:3| T75:3|T139:3|T203:3,  T27:0| T91:0|T155:0|T219:0,  T27:1| T91:1|T155:1|T219:1,  T27:2| T91:2|T155:2|T219:2,  T27:3| T91:3|T155:3|T219:3,  T43:0|T107:0|T171:0|T235:0,  T43:1|T107:1|T171:1|T235:1,  T43:2|T107:2|T171:2|T235:2,  T43:3|T107:3|T171:3|T235:3,  T59:0|T123:0|T187:0|T251:0,  T59:1|T123:1|T187:1|T251:1,  T59:2|T123:2|T187:2|T251:2,  T59:3|T123:3|T187:3|T251:3
// CHECK-MFMA: T12:0| T76:0|T140:0|T204:0,  T12:1| T76:1|T140:1|T204:1,  T12:2| T76:2|T140:2|T204:2,  T12:3| T76:3|T140:3|T204:3,  T28:0| T92:0|T156:0|T220:0,  T28:1| T92:1|T156:1|T220:1,  T28:2| T92:2|T156:2|T220:2,  T28:3| T92:3|T156:3|T220:3,  T44:0|T108:0|T172:0|T236:0,  T44:1|T108:1|T172:1|T236:1,  T44:2|T108:2|T172:2|T236:2,  T44:3|T108:3|T172:3|T236:3,  T60:0|T124:0|T188:0|T252:0,  T60:1|T124:1|T188:1|T252:1,  T60:2|T124:2|T188:2|T252:2,  T60:3|T124:3|T188:3|T252:3
// CHECK-MFMA: T13:0| T77:0|T141:0|T205:0,  T13:1| T77:1|T141:1|T205:1,  T13:2| T77:2|T141:2|T205:2,  T13:3| T77:3|T141:3|T205:3,  T29:0| T93:0|T157:0|T221:0,  T29:1| T93:1|T157:1|T221:1,  T29:2| T93:2|T157:2|T221:2,  T29:3| T93:3|T157:3|T221:3,  T45:0|T109:0|T173:0|T237:0,  T45:1|T109:1|T173:1|T237:1,  T45:2|T109:2|T173:2|T237:2,  T45:3|T109:3|T173:3|T237:3,  T61:0|T125:0|T189:0|T253:0,  T61:1|T125:1|T189:1|T253:1,  T61:2|T125:2|T189:2|T253:2,  T61:3|T125:3|T189:3|T253:3
// CHECK-MFMA: T14:0| T78:0|T142:0|T206:0,  T14:1| T78:1|T142:1|T206:1,  T14:2| T78:2|T142:2|T206:2,  T14:3| T78:3|T142:3|T206:3,  T30:0| T94:0|T158:0|T222:0,  T30:1| T94:1|T158:1|T222:1,  T30:2| T94:2|T158:2|T222:2,  T30:3| T94:3|T158:3|T222:3,  T46:0|T110:0|T174:0|T238:0,  T46:1|T110:1|T174:1|T238:1,  T46:2|T110:2|T174:2|T238:2,  T46:3|T110:3|T174:3|T238:3,  T62:0|T126:0|T190:0|T254:0,  T62:1|T126:1|T190:1|T254:1,  T62:2|T126:2|T190:2|T254:2,  T62:3|T126:3|T190:3|T254:3
// CHECK-MFMA: T15:0| T79:0|T143:0|T207:0,  T15:1| T79:1|T143:1|T207:1,  T15:2| T79:2|T143:2|T207:2,  T15:3| T79:3|T143:3|T207:3,  T31:0| T95:0|T159:0|T223:0,  T31:1| T95:1|T159:1|T223:1,  T31:2| T95:2|T159:2|T223:2,  T31:3| T95:3|T159:3|T223:3,  T47:0|T111:0|T175:0|T239:0,  T47:1|T111:1|T175:1|T239:1,  T47:2|T111:2|T175:2|T239:2,  T47:3|T111:3|T175:3|T239:3,  T63:0|T127:0|T191:0|T255:0,  T63:1|T127:1|T191:1|T255:1,  T63:2|T127:2|T191:2|T255:2,  T63:3|T127:3|T191:3|T255:3


// CHECK-HW: Warp0:
// CHECK-HW: Warp1:
// CHECK-HW: Warp2:
// CHECK-HW: Warp3:
