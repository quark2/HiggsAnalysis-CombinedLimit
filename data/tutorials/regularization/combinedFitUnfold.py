import ROOT, math, numpy
import os, sys
import subprocess


dAlphaDsc = 1.5

listAllRowRegularization = [
    [],               # No type - just for dummy
    [1.0],            # Type 1: Just norm
    [1.0, -1.0],      # Type 2: Making first derivatives smooth
    [1.0, -2.0, 1.0], # Type 3: Making second derivatives smooth
  ]

strCardContentTemplate = """imax %(binnum)i
jmax %(procnum)i
kmax *
---------------
shapes * * %(inputroot)s hist_$CHANNEL_$PROCESS hist_$CHANNEL_$PROCESS_sysunc$SYSTEMATIC
---------------
bin %(listbin)s
observation %(listnumdata)s
------------------------------
bin      %(expandlistbin)s
process  %(listproc)s
process  %(listidxproc)s
rate     %(listnumproc)s
--------------------------------
%(rateParams)s%(listConstr)s%(BBLOption)s%(listunc)s"""

strLineRateParamSigTemplate = "%s rateParam * *_SigRecInGen%02i %lf [0.0,%lf]"
strLineRateParamBkgTemplate = "%s rateParam %s %s 1.0 [0.0,10.0]"

strLineConstrParamBkgTemplate = "%s param 1.0 0.3"
strLineConstrRegularizationTemplate = "constrReg%02i constr %s delta[%lf]"

strParamForCardTemplate = "--PO 'map=%s:%s[%lf,0.0,%lf]'"
strParamForRunTemplate = "-P %s"

strCmdForWorkspaceTemplate = "text2workspace.py " +\
    "-P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel %(paramsTxt)s %(filename)s"

# verbose level 3: Nuisance parameters come out
strCmdForRunTemplate = "combine %(filename)s -M MultiDimFit %(paramsCombine)s " +\
    "--verbose 3 --saveNLL --saveSpecifiedNuis=all --cminFinalHesse 1"


# Extracting a first float number in a given text
def GetFirstFloat(strIn): 
  nIdx = 0
  nIdxFirst = -1
  
  while True: 
    if nIdxFirst < 0: 
      if nIdx >= len(strIn): return (None, strIn)
      if strIn[ nIdx ].isdigit(): nIdxFirst = nIdx
    else: 
      try: float(strIn[ nIdxFirst:nIdx ])
      except ValueError: break
      if nIdx >= len(strIn): break
    
    nIdx += 1
  
  # Considering sign
  if nIdxFirst > 0 and strIn[ nIdxFirst - 1 ] == "-": nIdxFirst -= 1
  
  return (float(strIn[ nIdxFirst:( nIdx - 1 ) ]), strIn[ nIdx: ])


def FuncResponseMatrix(dX, dY): 
  return 0.5 / 36.0 * ( 1.0 / ( 1.0 + dY ** 4 ) ) * ( 1.0 / ( 1.0 + ( abs(dY - dX) ** 0.5 ) ** 2 ) )


def FuncDiscrSig(dX): 
  dZ = 0.5 * ( 1.0 + dX )
  return dZ ** dAlphaDsc


def FuncDiscrBkg(dX): 
  dZ = 0.5 * ( 1.0 - dX )
  return dZ ** dAlphaDsc


def GenerateResponseMatrix(nNBinGen, nNBinRec, nNBinDsc): 
  h3RespMat = ROOT.TH3F("hist3D_RespMat", "Response matrix", nNBinGen, -1, 1, nNBinRec, -1, 1, nNBinDsc, -1, 1)
  
  dNor = 1.0 / sum([ FuncDiscrSig(h3RespMat.GetZaxis().GetBinCenter(k)) for k in range(1, nNBinDsc + 1) ])
  
  for i in range(1, nNBinGen + 1): 
    for j in range(1, nNBinRec + 1): 
      for k in range(1, nNBinDsc + 1): 
        dX = h3RespMat.GetXaxis().GetBinCenter(i)
        dY = h3RespMat.GetYaxis().GetBinCenter(j)
        dZ = h3RespMat.GetZaxis().GetBinCenter(k)
        h3RespMat.SetBinContent(i, j, k, FuncResponseMatrix(dX, dY) * FuncDiscrSig(dZ) * dNor)
  
  return h3RespMat


def GenerateSigGen(nNBinGen, dXSec): 
  hSigGen = ROOT.TH1D("hist_SigGen", "Signal target var in gen lvl", nNBinGen, -1, 1)
  
  for i in range(1, nNBinGen + 1): 
    dX = hSigGen.GetXaxis().GetBinCenter(i) + 1.0
    hSigGen.SetBinContent(i, dXSec * dX * math.exp(-2 * dX * dX))
    hSigGen.SetBinError(i, hSigGen.GetBinContent(i) ** 0.5)
  
  return hSigGen


def SimulateSigRec(h3RespMat, hSigGen):  # !!! Simulate...???
  nNBinGen = h3RespMat.GetNbinsX()
  nNBinRec = h3RespMat.GetNbinsY()
  nNBinDsc = h3RespMat.GetNbinsZ()
  
  hSigRec = ROOT.TH2D("hist_SigRec", "Signal target var in reco lvl", nNBinRec, -1, 1, nNBinDsc, -1, 1)
  
  for i in range(1, nNBinDsc + 1): 
    for j in range(1, nNBinRec + 1): 
      dVal = sum([ h3RespMat.GetBinContent(k, j, i) * hSigGen.GetBinContent(k) for k in range(1, nNBinGen + 1) ])
      hSigRec.SetBinContent(j, i, dVal)
      hSigRec.SetBinError(j, i, dVal ** 0.5)
  
  return hSigRec


def SimulateBkgRec(nNBinRec, nNBinDsc, dXSec, hBkgRecOrg = None, dDistortion = 0.0): 
  hBkgRec = None
  if hBkgRecOrg == None: 
    hBkgRec = ROOT.TH2D("hist_BkgRec", "Background target var in reco lvl", nNBinRec, -1, 1, nNBinDsc, -1, 1)
  else: 
    hBkgRec = hBkgRecOrg
  
  dNor = 1.0 / sum([ FuncDiscrBkg(hBkgRec.GetYaxis().GetBinCenter(k)) for k in range(1, nNBinDsc + 1) ])
  
  for i in range(1, nNBinRec + 1): 
    for j in range(1, nNBinDsc + 1): 
      dX = hBkgRec.GetXaxis().GetBinCenter(i)
      dY = hBkgRec.GetYaxis().GetBinCenter(j)
      
      dVal = dXSec * dNor * FuncDiscrBkg(dY) * ( 9.33 - 4.31 * dX - 7.29 * ( dX ** 2 ) + 3.21 * ( dX ** 3 ) )
      dVal *= ( 1.0 + ( dDistortion * dY ) ** 2 ) ** 2
      hBkgRec.SetBinContent(i, j, dVal)
      hBkgRec.SetBinError(i, j, dVal ** 0.5 * 10.0)
  
  return hBkgRec


def GeneratePseudoData(hSigRec, fSFSig = 1.0, hBkgRec = None, fSFBkg = 1.0): 
  nNBinRec = hSigRec.GetNbinsX()
  nNBinDsc = hSigRec.GetNbinsY()
  
  hData = ROOT.TH2D("hist_data_obs", "Data target var in reco lvl", nNBinRec, -1, 1, nNBinDsc, -1, 1)
  
  hData.Add(hSigRec, fSFSig)
  if hBkgRec != None: hData.Add(hBkgRec, fSFBkg)
  
  return hData


def GetSliceInOneObsBinForRespMat(h3RespMat): 
  listHist = []
  
  nNBinGen = h3RespMat.GetNbinsX()
  nNBinRec = h3RespMat.GetNbinsY()
  
  for i in range(1, nNBinGen + 1): 
    listHist.append([])
    for j in range(1, nNBinRec + 1): 
      listHist[ -1 ].append(h3RespMat.ProjectionZ("hist_slice%02i_slice%02i_SigRecInGen%02i"%(j, j, i), i, i, j, j))
  
  return listHist


def GetSliceInOneObsBinFor2DRec(hRec, bDoubleSliceLabel = False): 
  listHist = []
  
  nNBinRec = hRec.GetNbinsX()
  
  for i in range(1, nNBinRec + 1): 
    strName = hRec.GetName().replace("hist_", "hist_slice%02i_"%i + ( "slice%02i_"%i if bDoubleSliceLabel else "" ))
    listHist.append(hRec.ProjectionY(strName, i, i))
  
  return listHist


def CreateHistFile(strFilename, listHistResp, listHistData, listHistSigRec, listHistBkgRec = []): 
  fHist = ROOT.TFile(strFilename, "RECREATE")
  
  fHist.cd()
  
  for l in listHistResp: 
    for h in l: h.Write()
  
  for h in listHistData:   h.Write()
  #hSigRec.Write()
  for h in listHistBkgRec: h.Write()
  
  fHist.Close()


def AddRegularizationTerm(listParamPOI, hBias, fDelta, nType = 3): 
  nNBinGen = len(listParamPOI)
  
  listEntry = []
  listRowReg = listAllRowRegularization[ nType ]
  nNDim = nNBinGen - len(listRowReg) + 1
  
  for i in range(nNDim): 
    listTerm = []
    listParamUsed = []
    
    for j, fRegFactor in enumerate(listRowReg): 
      strNameParam = listParamPOI[ i + j ]
      fValBias = hBias.GetBinContent(i + j + 1)
      listTerm.append("(%0.1lf)*(%s-%lf)"%(fRegFactor, strNameParam, fValBias))
      listParamUsed.append(strNameParam)
    
    strAllTerm = "+".join(listTerm)
    listEntry.append([ strAllTerm, ",".join(listParamUsed) ])
  
  return listEntry


def CreateWSCard(strFilename, strFileRoot, hSigGen, fDelta, listRegTerms, 
                 listHistResp, listHistData, listHistSigRec, listHistBkgRec = []): 
  nNBinGen = len(listHistResp)
  nNBinRec = len(listHistData)
  nNBinDsc = listHistData[ -1 ].GetNbinsX()
  
  listColData = []
  listColProc = []
  
  listNamePOI = []
  listRateParams = []
  listConstr = []
  
  listParamWS  = []
  listParamRun = []
  
  nNumProc = 0
  nIdxProcSig = 0
  nIdxProcBkg = 1
  
  for nIdxRec in range(nNBinRec): 
    listColData.append({"bin": "slice%02i"%(nIdxRec + 1), "rate": listHistData[ nIdxRec ].Integral()})
    
    for nIdxGen in range(nNBinGen): 
      histCurr = listHistResp[ nIdxGen ][ nIdxRec ]
      strNameProc = histCurr.GetName().replace("hist_slice%02i_"%(nIdxRec + 1), "")
      strNameBin = "slice%02i"%(nIdxRec + 1)
      
      listColProc.append({"bin": strNameBin, "proc": strNameProc, 
        "procidx": nIdxProcSig, "rate": histCurr.Integral()})
      
      nNumProc += 1
      nIdxProcSig -= 1
    
    for listBkg in listHistBkgRec: 
      histCurr = listBkg[ nIdxRec ]
      strNameProc = histCurr.GetName().replace("hist_slice%02i_"%(nIdxRec + 1), "")
      strNameBin = "slice%02i"%(nIdxRec + 1)
      
      listColProc.append({"bin": strNameBin, "proc": strNameProc, 
        "procidx": nIdxProcBkg, "rate": histCurr.Integral()})
      
      listRateParams.append(strLineRateParamBkgTemplate%(strNameProc, strNameBin, strNameProc))
      listConstr.append(strLineConstrParamBkgTemplate%(strNameProc))
      
      nNumProc += 1
      nIdxProcBkg += 1
  
  for i in range(1, nNBinGen + 1): 
    strNameParam = "unfold_%i"%i
    fValGen = hSigGen.GetBinContent(i)
    
    listNamePOI.append(strNameParam)
    listRateParams.append(strLineRateParamSigTemplate%(strNameParam, i, fValGen, fValGen * 2.0))
    
    listParamWS.append(strParamForCardTemplate%(strNameParam, strNameParam, fValGen, fValGen * 2.0))
    listParamRun.append(strParamForRunTemplate%(strNameParam))
  
  listRegPre = AddRegularizationTerm(listNamePOI, hSigGen, fDelta, 3)
  for l in listRegPre: listRegTerms.append(l[ 0 ])
  listEntry = [ "(%s) {%s}"%(l[ 0 ], l[ 1 ]) for l in listRegPre ]
  listConstr += [ strLineConstrRegularizationTemplate%(i + 1, s, fDelta) for i, s in enumerate(listEntry) ]
  
  dicIn = {}
  
  dicIn[ "inputroot" ]     = strFileRoot
  dicIn[ "binnum" ]        = nNBinRec
  dicIn[ "procnum" ]       = nNumProc - 1
  
  dicIn[ "listbin" ]       = "  ".join([      d[ "bin" ]     for d in listColData ])
  dicIn[ "listnumdata" ]   = "  ".join([ "%f"%d[ "rate" ]    for d in listColData ])
  
  dicIn[ "expandlistbin" ] = "  ".join([      d[ "bin" ]     for d in listColProc ])
  dicIn[ "listproc" ]      = "  ".join([      d[ "proc" ]    for d in listColProc ])
  dicIn[ "listidxproc" ]   = "  ".join([ "%i"%d[ "procidx" ] for d in listColProc ])
  dicIn[ "listnumproc" ]   = "  ".join([ "%f"%d[ "rate" ]    for d in listColProc ])
  
  dicIn[ "rateParams" ]    = "\n".join(listRateParams + ( [ "" ] if len(listRateParams) > 0 else [] ))
  dicIn[ "listConstr" ]    = "\n".join(listConstr + ( [ "" ] if len(listConstr) > 0 else [] ))
  
  dicIn[ "BBLOption" ]     = "* autoMCStats 0\n"
  dicIn[ "listunc" ]       = "" #"\n".join(listUnc)
  
  strCardContent = strCardContentTemplate%dicIn
  
  with open(strFilename, "w") as fCard: fCard.write(strCardContent)
  
  return ([ " ".join(listParamWS), " ".join(listParamRun) ], listNamePOI)


def InterpretRes(listLineRes, nNBinGen): 
  listVal = []
  listErr = []
  
  for i in range(1, nNBinGen + 1): 
    strNameBin = "unfold_%i"%i
    for s in listLineRes: 
      if s.startswith(strNameBin) and "+/-" in s: 
        s = s.replace(strNameBin, "")
        (fVal, s) = GetFirstFloat(s)
        (fErr, s) = GetFirstFloat(s)
        
        listVal.append(fVal)
        listErr.append(fErr)
  
  hSigUnfold = ROOT.TH1D("hSigUnfold", "", len(listVal), -1, 1)
  for i, dV in enumerate(listVal): hSigUnfold.SetBinContent(i + 1, dV)
  for i, dE in enumerate(listErr): hSigUnfold.SetBinError(i + 1, dE)
  
  return (listVal, listErr, hSigUnfold)


def CalcRegularizationTerm(listRegTerms, listParamPOI, listVal): 
  listEvalTerm = []
  
  for strTerm in listRegTerms: 
    strTermEval = strTerm
    for i in range(len(listParamPOI)): strTermEval = strTermEval.replace(listParamPOI[ i ], "%lf"%listVal[ i ])
    listEvalTerm.append(eval(strTermEval))
  
  return math.log(sum([ d ** 2 for d in listEvalTerm ]))


def ReadNLL(listOut): 
  for s in listOut: 
    if s.startswith("FVAL"): 
      (fNLL, s) = GetFirstFloat(s)
      return fNLL


def ReadHesseMatrixOnlyPOI(listOut, listNamePOI): 
  nNBinGen = len(listNamePOI)
  
  nIdxStart   = 0
  nIdxEndMtx  = 0
  nIdxEnd     = 0
  
  for i, s in enumerate(listOut): 
    if s.startswith("MnUserCovariance Parameter correlations"): 
      nIdxStart = i
      break
  
  for i, s in enumerate(listOut): 
    if s.startswith("MnUserCovariance:"): 
      nIdxEndMtx = i
      break
  
  for i, s in enumerate(listOut): 
    if s.startswith("external parameters"): 
      nIdxEnd = i
      break
  
  listAllMatrix = listOut[ nIdxStart + 1 : nIdxEndMtx ]
  listAllEntry  = listOut[ nIdxEndMtx    : nIdxEnd    ]
  listAllMatrix = [ s for s in listAllMatrix if len(s.split()) > 0 ]
  
  listAllMatrix.reverse()
  listAllEntry.reverse()
  
  listIdxPOI = []
  
  for strNamePOI in listNamePOI: 
    for s in listAllEntry: 
      if strNamePOI in s: 
        listIdxPOI.append(int(s.split()[ 0 ]))
        break
  
  listCovMtx = []
  nIdxIni = len(listAllMatrix) - len(listNamePOI)
  nIdxFin = len(listAllMatrix)
  
  for nIdxX in listIdxPOI: 
    listEntries = listAllMatrix[ nIdxX ].split()
    listCovMtx.append([ float(listEntries[ nIdxY ]) for nIdxY in listIdxPOI ])
  
  matCov = numpy.array(listCovMtx)
  matInvCov = numpy.linalg.inv(matCov)
  
  #print(matCov)
  #print(matInvCov)
  
  listRho = [ ( 1.0 - 1.0 / ( matCov[ i ][ i ] * matInvCov[ i ][ i ] ) ) ** 0.5 for i in range(nNBinGen) ]
  fRhoMax = sum(listRho) / len(listRho)
  #fRhoMax = max(listRho)
  
  return fRhoMax


fDelta = float(sys.argv[ 1 ])

nNBinGen = 8
nNBinRec = 16
nNBinDsc = 6

dXSecSig = 800000
dXSecBkgRec = 700

strFileHist = "asdf.root"
strFileCard = "asdfcard.txt"

listRegTerms = []

h3RespMat = GenerateResponseMatrix(nNBinGen, nNBinRec, nNBinDsc)

hSigGen = GenerateSigGen(nNBinGen, dXSecSig)
hSigRec = SimulateSigRec(h3RespMat, hSigGen)
hBkgRec = SimulateBkgRec(nNBinRec, nNBinDsc, dXSecBkgRec)

hData = GeneratePseudoData(hSigRec, 1.0, hBkgRec, 1.1)
hBkgRec = SimulateBkgRec(nNBinRec, nNBinDsc, dXSecBkgRec, hBkgRec, 0.3)

listHistData   = GetSliceInOneObsBinFor2DRec(hData)
listHistSigRec = GetSliceInOneObsBinFor2DRec(hSigRec, True)
listHistBkgRec = GetSliceInOneObsBinFor2DRec(hBkgRec, True)

listHistAllBkgRec = [ l for l in [ listHistBkgRec ] if len(l) > 0 ]

listHistResp = GetSliceInOneObsBinForRespMat(h3RespMat)
CreateHistFile(strFileHist, listHistResp, listHistData, hSigRec, listHistBkgRec)

(listParam, listParamPOI) = CreateWSCard(strFileCard, strFileHist, hSigGen, fDelta, listRegTerms, 
                                        listHistResp, listHistData, listHistSigRec, listHistAllBkgRec)

listCmd = []

os.environ[ "CMSSW_BASE" ] = os.path.realpath(__file__).rsplit("/src/", 1)[ 0 ]
listCmd += [ "source /opt/root/6.12.06_withpy2/bin/thisroot.sh",
    "source " + os.path.join(os.environ[ "CMSSW_BASE" ],
    "src/nano/analysis/test/singletop/env_standalone_mine.sh") ]

listCmd.append(strCmdForWorkspaceTemplate%{"paramsTxt": listParam[ 0 ], "filename": strFileCard})
listCmd.append(strCmdForRunTemplate%{"paramsCombine": listParam[ 1 ], 
    "filename": strFileCard.replace(".txt", ".root")} + " | iconv --to-code utf-8//IGNORE")

#print(" ; ".join(listCmd))
listOutput = subprocess.getoutput(" ; ".join(listCmd)).splitlines()
#for s in listOutput: print(s)

listOutput.reverse()

fRhoMax = ReadHesseMatrixOnlyPOI(listOutput, listParamPOI)
print(fDelta, fRhoMax)

(listResVal, listResErr, hSigUnfold) = InterpretRes(listOutput, nNBinGen)
print(listResVal, listResErr)
print([ hSigGen.GetBinContent(i) for i in range(1, nNBinGen + 1) ], [ hSigGen.GetBinError(i) for i in range(1, nNBinGen + 1) ])
#for i in range(1, nNBinGen + 1): print("%lf +/- %lf"%(hSigGen.GetBinContent(i), hSigGen.GetBinError(i)))

fNLL = ReadNLL(listOutput)
fL2 = CalcRegularizationTerm(listRegTerms, listParamPOI, listResVal)

#print(1.0 / ( fDelta * fDelta ), fNLL - fL2, fL2)


