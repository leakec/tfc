(* Content-type: application/vnd.wolfram.cdf.text *)

(*** Wolfram CDF File ***)
(* http://www.wolfram.com/cdf *)

(* CreatedBy='Mathematica 12.0' *)

(***************************************************************************)
(*                                                                         *)
(*                                                                         *)
(*  Under the Wolfram FreeCDF terms of use, this file and its content are  *)
(*  bound by the Creative Commons BY-SA Attribution-ShareAlike license.    *)
(*                                                                         *)
(*        For additional information concerning CDF licensing, see:        *)
(*                                                                         *)
(*         www.wolfram.com/cdf/adopting-cdf/licensing-options.html         *)
(*                                                                         *)
(*                                                                         *)
(***************************************************************************)

(*CacheID: 234*)
(* Internal cache information:
NotebookFileLineBreakTest
NotebookFileLineBreakTest
NotebookDataPosition[      1088,         20]
NotebookDataLength[     14407,        407]
NotebookOptionsPosition[     12446,        364]
NotebookOutlinePosition[     12809,        380]
CellTagsIndexPosition[     12766,        377]
WindowFrame->Normal*)

(* Beginning of Notebook Content *)
Notebook[{
Cell["Chapter 1 Problem 2:", "Subsubsection",
 CellChangeTimes->{{3.820586254287455*^9, 3.820586272280244*^9}, {
  3.8205879464678373`*^9, 
  3.820587946565063*^9}},ExpressionUUID->"699b74c8-e698-404e-8ffd-\
7bdebbfbe347"],

Cell[CellGroupData[{

Cell["1.", "Subsubsection",
 CellChangeTimes->{{3.8205864936097603`*^9, 
  3.820586499876956*^9}},ExpressionUUID->"625fcf49-195a-485a-89d1-\
d3f697221d08"],

Cell["Constrained expression", "Text",
 CellChangeTimes->{{3.82058627523597*^9, 3.820586277701434*^9}, {
  3.820586345940712*^9, 3.820586347755028*^9}, {3.8205864717813807`*^9, 
  3.820586496739429*^9}},ExpressionUUID->"bb869408-732e-44bb-9089-\
813d400fe162"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"y1", "[", "x_", "]"}], ":=", 
   RowBox[{
    RowBox[{"g", "[", "x", "]"}], "+", 
    RowBox[{"(", 
     RowBox[{"y0", "-", 
      RowBox[{"g", "[", "x0", "]"}]}], ")"}], "+", 
    RowBox[{
     RowBox[{"(", 
      RowBox[{
       RowBox[{
        RowBox[{"Sec", "[", "x0", "]"}], " ", 
        RowBox[{"Sin", "[", "x", "]"}]}], "-", 
       RowBox[{"Tan", "[", "x0", "]"}]}], ")"}], 
     RowBox[{"(", 
      RowBox[{"yx0", "-", 
       RowBox[{"(", 
        RowBox[{
         RowBox[{"D", "[", 
          RowBox[{
           RowBox[{"g", "[", "x", "]"}], ",", "x"}], "]"}], "/.", 
         RowBox[{"x", "\[Rule]", "x0"}]}], ")"}]}], ")"}]}]}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"y1", "[", "x", "]"}], "//", "TraditionalForm"}]}], "Input",
 CellChangeTimes->{{3.8205862854499063`*^9, 3.8205863410355873`*^9}, {
  3.820586458036991*^9, 3.820586460723352*^9}, {3.820587569073257*^9, 
  3.820587591257009*^9}, {3.820587723983576*^9, 3.820587744850932*^9}, {
  3.825094746529686*^9, 3.8250947491773453`*^9}},
 CellLabel->"In[1]:=",ExpressionUUID->"4356ffef-aa16-4ed7-a16e-04cbef825621"],

Cell[BoxData[
 FormBox[
  RowBox[{
   RowBox[{
    RowBox[{"(", 
     RowBox[{"yx0", "-", 
      RowBox[{
       SuperscriptBox["g", "\[Prime]",
        MultilineFunction->None], "(", "x0", ")"}]}], ")"}], " ", 
    RowBox[{"(", 
     RowBox[{
      RowBox[{
       RowBox[{"sin", "(", "x", ")"}], " ", 
       RowBox[{"sec", "(", "x0", ")"}]}], "-", 
      RowBox[{"tan", "(", "x0", ")"}]}], ")"}]}], "+", 
   RowBox[{"g", "(", "x", ")"}], "-", 
   RowBox[{"g", "(", "x0", ")"}], "+", "y0"}], TraditionalForm]], "Output",
 CellChangeTimes->{{3.8250947499271584`*^9, 3.825094769634039*^9}, 
   3.829829929804879*^9},
 CellLabel->
  "Out[2]//TraditionalForm=",ExpressionUUID->"6887eaf4-a531-4fd6-bae7-\
ab6d021cde57"]
}, Open  ]],

Cell["Check the constraints", "Text",
 CellChangeTimes->{{3.820586351355624*^9, 
  3.820586356156693*^9}},ExpressionUUID->"d889e916-a4db-4c04-9f2d-\
93d823b43940"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"y1", "[", "x0", "]"}], "-", "y0"}], " ", "\[Equal]", " ", 
  "0"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"(", 
    RowBox[{
     RowBox[{"D", "[", 
      RowBox[{
       RowBox[{"y1", "[", "x", "]"}], ",", "x"}], "]"}], "/.", 
     RowBox[{"x", "\[Rule]", "x0"}]}], ")"}], "-", "yx0"}], " ", "\[Equal]", 
  " ", "0"}]}], "Input",
 CellChangeTimes->{{3.820586368649537*^9, 3.820586402336206*^9}, 
   3.820586534390914*^9},
 CellLabel->"In[3]:=",ExpressionUUID->"0112db79-82b2-40a5-9583-c1b7738c4c4c"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.820586396556509*^9, 3.820586429999325*^9, 3.8205864632424583`*^9, 
   3.820587137977333*^9, {3.820587573394577*^9, 3.820587593834496*^9}, 
   3.820587748196183*^9, 3.8205879522805853`*^9, 3.825094769729347*^9, 
   3.829829929973606*^9},
 CellLabel->"Out[3]=",ExpressionUUID->"d88d71cf-5fd8-45c1-95b1-2e73e9a087be"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.820586396556509*^9, 3.820586429999325*^9, 3.8205864632424583`*^9, 
   3.820587137977333*^9, {3.820587573394577*^9, 3.820587593834496*^9}, 
   3.820587748196183*^9, 3.8205879522805853`*^9, 3.825094769729347*^9, 
   3.829829929977117*^9},
 CellLabel->"Out[4]=",ExpressionUUID->"38aa5c2a-e135-4b55-b9d4-484ad3e27afa"]
}, Open  ]]
}, Open  ]],

Cell[CellGroupData[{

Cell["2.", "Subsubsection",
 CellChangeTimes->{{3.820586506124516*^9, 
  3.820586506313881*^9}},ExpressionUUID->"bde8c775-c83a-4510-b0ca-\
ff041ee2a122"],

Cell["Constrained expression", "Text",
 CellChangeTimes->{{3.820586467388218*^9, 
  3.820586503892722*^9}},ExpressionUUID->"0fc09061-e01e-4f69-8870-\
8c827fc41dff"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"y2", "[", "x_", "]"}], ":=", 
   RowBox[{
    RowBox[{"g", "[", "x", "]"}], "+", 
    RowBox[{"(", 
     RowBox[{"yf", "-", 
      RowBox[{"g", "[", "xf", "]"}]}], ")"}], "+", 
    RowBox[{
     RowBox[{"Sec", "[", "x0", "]"}], " ", 
     RowBox[{"(", 
      RowBox[{
       RowBox[{"Sin", "[", "x", "]"}], "-", 
       RowBox[{"Sin", "[", "xf", "]"}]}], ")"}], 
     RowBox[{"(", 
      RowBox[{"yx0", "-", 
       RowBox[{"(", 
        RowBox[{
         RowBox[{"D", "[", 
          RowBox[{
           RowBox[{"g", "[", "x", "]"}], ",", "x"}], "]"}], "/.", 
         RowBox[{"x", "\[Rule]", "x0"}]}], ")"}]}], ")"}]}]}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"y2", "[", "x", "]"}], "//", "TraditionalForm"}]}], "Input",
 CellChangeTimes->{{3.820586523827012*^9, 3.820586545243299*^9}, {
   3.820587601982132*^9, 3.8205876036263933`*^9}, {3.8205876443603897`*^9, 
   3.820587647116292*^9}, 3.820587777089287*^9, {3.8250947542735453`*^9, 
   3.8250947568809443`*^9}},
 CellLabel->"In[5]:=",ExpressionUUID->"3043dc14-1a85-4df4-b63a-8c8bf3dea910"],

Cell[BoxData[
 FormBox[
  RowBox[{
   RowBox[{
    RowBox[{"sec", "(", "x0", ")"}], " ", 
    RowBox[{"(", 
     RowBox[{
      RowBox[{"sin", "(", "x", ")"}], "-", 
      RowBox[{"sin", "(", "xf", ")"}]}], ")"}], " ", 
    RowBox[{"(", 
     RowBox[{"yx0", "-", 
      RowBox[{
       SuperscriptBox["g", "\[Prime]",
        MultilineFunction->None], "(", "x0", ")"}]}], ")"}]}], "+", 
   RowBox[{"g", "(", "x", ")"}], "-", 
   RowBox[{"g", "(", "xf", ")"}], "+", "yf"}], TraditionalForm]], "Output",
 CellChangeTimes->{3.825094769753489*^9, 3.8298299300105886`*^9},
 CellLabel->
  "Out[6]//TraditionalForm=",ExpressionUUID->"26e12a7b-159a-46a8-bfcc-\
eec29e629dc2"]
}, Open  ]],

Cell["Check the constraints", "Text",
 CellChangeTimes->{{3.820586531588002*^9, 
  3.8205865321666183`*^9}},ExpressionUUID->"e63e1c12-a1fc-401c-a1b1-\
1fe3210d22c1"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"(", 
    RowBox[{
     RowBox[{"D", "[", 
      RowBox[{
       RowBox[{"y2", "[", "x", "]"}], ",", "x"}], "]"}], "/.", 
     RowBox[{"x", "\[Rule]", "x0"}]}], ")"}], "-", "yx0"}], " ", "\[Equal]", 
  " ", "0"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"y2", "[", "xf", "]"}], "-", "yf"}], " ", "\[Equal]", " ", 
  "0"}]}], "Input",
 CellChangeTimes->{{3.820586557391293*^9, 3.820586571712685*^9}},
 CellLabel->"In[7]:=",ExpressionUUID->"918474b6-3d1d-43e5-9c80-78d4b4b819a0"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{3.820586578018466*^9, 3.8205871380216227`*^9, 
  3.820587605377068*^9, 3.820587785537236*^9, 3.820587952363559*^9, 
  3.8250947697880287`*^9, 3.829829930038877*^9},
 CellLabel->"Out[7]=",ExpressionUUID->"68bb4015-4c4e-4dc0-80f6-3ea4eed4a223"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{3.820586578018466*^9, 3.8205871380216227`*^9, 
  3.820587605377068*^9, 3.820587785537236*^9, 3.820587952363559*^9, 
  3.8250947697880287`*^9, 3.829829930040908*^9},
 CellLabel->"Out[8]=",ExpressionUUID->"b21d99d6-c5bf-49e8-aded-c3f8281d8c78"]
}, Open  ]]
}, Open  ]],

Cell[CellGroupData[{

Cell["3.", "Subsubsection",
 CellChangeTimes->{{3.8205865811403313`*^9, 
  3.820586581400208*^9}},ExpressionUUID->"422a493f-06cb-40a4-8f1e-\
03d08b4f607c"],

Cell["Constrained expression", "Text",
 CellChangeTimes->{
  3.820586588142373*^9},ExpressionUUID->"2e2ed5b2-aeb4-4f08-a856-\
630ac1ca4c90"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"y3", "[", "x_", "]"}], ":=", 
   RowBox[{
    RowBox[{"g", "[", "x", "]"}], "+", 
    RowBox[{"(", 
     RowBox[{"y0", "-", 
      RowBox[{"g", "[", "x0", "]"}]}], ")"}], "+", 
    RowBox[{
     RowBox[{"Sec", "[", "xf", "]"}], " ", 
     RowBox[{"(", 
      RowBox[{
       RowBox[{"Sin", "[", "x", "]"}], "-", 
       RowBox[{"Sin", "[", "x0", "]"}]}], ")"}], 
     RowBox[{"(", 
      RowBox[{"yxf", "-", 
       RowBox[{"(", 
        RowBox[{
         RowBox[{"D", "[", 
          RowBox[{
           RowBox[{"g", "[", "x", "]"}], ",", "x"}], "]"}], "/.", 
         RowBox[{"x", "\[Rule]", "xf"}]}], ")"}]}], ")"}]}]}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"y3", "[", "x", "]"}], "//", "TraditionalForm"}]}], "Input",
 CellChangeTimes->{{3.820586594811557*^9, 3.820586644212625*^9}, 
   3.820587810450382*^9, {3.825094760289098*^9, 3.8250947630166903`*^9}},
 CellLabel->"In[9]:=",ExpressionUUID->"2ad49d71-dd74-45b0-908d-0ce3169206f4"],

Cell[BoxData[
 FormBox[
  RowBox[{
   RowBox[{
    RowBox[{"sec", "(", "xf", ")"}], " ", 
    RowBox[{"(", 
     RowBox[{
      RowBox[{"sin", "(", "x", ")"}], "-", 
      RowBox[{"sin", "(", "x0", ")"}]}], ")"}], " ", 
    RowBox[{"(", 
     RowBox[{"yxf", "-", 
      RowBox[{
       SuperscriptBox["g", "\[Prime]",
        MultilineFunction->None], "(", "xf", ")"}]}], ")"}]}], "+", 
   RowBox[{"g", "(", "x", ")"}], "-", 
   RowBox[{"g", "(", "x0", ")"}], "+", "y0"}], TraditionalForm]], "Output",
 CellChangeTimes->{3.8250947698242073`*^9, 3.829829930078828*^9},
 CellLabel->
  "Out[10]//TraditionalForm=",ExpressionUUID->"90a1576c-6a0c-4184-b709-\
62f06a70712f"]
}, Open  ]],

Cell["Check the constraints", "Text",
 CellChangeTimes->{{3.8205866255269947`*^9, 
  3.820586626855633*^9}},ExpressionUUID->"112ebb53-6b0b-4781-b847-\
ee3a1450365b"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"y3", "[", "x0", "]"}], "-", "y0"}], " ", "\[Equal]", " ", 
  "0"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"(", 
    RowBox[{
     RowBox[{"D", "[", 
      RowBox[{
       RowBox[{"y3", "[", "x", "]"}], ",", "x"}], "]"}], "/.", 
     RowBox[{"x", "\[Rule]", "xf"}]}], ")"}], "-", "yxf"}], " ", "\[Equal]", 
  " ", "0"}]}], "Input",
 CellChangeTimes->{{3.82058663440956*^9, 3.8205866412946568`*^9}},
 CellLabel->"In[11]:=",ExpressionUUID->"64f13582-4f3d-4b5a-a582-687056326408"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{3.820586646760118*^9, 3.820587138064991*^9, 
  3.820587812773291*^9, 3.820587952421236*^9, 3.825094769854781*^9, 
  3.829829930087421*^9},
 CellLabel->"Out[11]=",ExpressionUUID->"8a2a743e-75af-4c47-abbd-0885c7e6e7c7"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{3.820586646760118*^9, 3.820587138064991*^9, 
  3.820587812773291*^9, 3.820587952421236*^9, 3.825094769854781*^9, 
  3.8298299300894823`*^9},
 CellLabel->"Out[12]=",ExpressionUUID->"a1227b00-99f1-4653-b22d-a4fb81ae818e"]
}, Open  ]]
}, Open  ]]
},
WindowSize->{1360, 704},
WindowMargins->{{0, Automatic}, {0, Automatic}},
Magnification:>1.6 Inherited,
FrontEndVersion->"12.0 for Linux x86 (64-bit) (April 8, 2019)",
StyleDefinitions->"Default.nb"
]
(* End of Notebook Content *)

(* Internal cache information *)
(*CellTagsOutline
CellTagsIndex->{}
*)
(*CellTagsIndex
CellTagsIndex->{}
*)
(*NotebookFileOutline
Notebook[{
Cell[1488, 33, 222, 4, 73, "Subsubsection",ExpressionUUID->"699b74c8-e698-404e-8ffd-7bdebbfbe347"],
Cell[CellGroupData[{
Cell[1735, 41, 155, 3, 58, "Subsubsection",ExpressionUUID->"625fcf49-195a-485a-89d1-d3f697221d08"],
Cell[1893, 46, 260, 4, 56, "Text",ExpressionUUID->"bb869408-732e-44bb-9089-813d400fe162"],
Cell[CellGroupData[{
Cell[2178, 54, 1153, 31, 85, "Input",ExpressionUUID->"4356ffef-aa16-4ed7-a16e-04cbef825621"],
Cell[3334, 87, 716, 21, 78, "Output",ExpressionUUID->"6887eaf4-a531-4fd6-bae7-ab6d021cde57"]
}, Open  ]],
Cell[4065, 111, 163, 3, 56, "Text",ExpressionUUID->"d889e916-a4db-4c04-9f2d-93d823b43940"],
Cell[CellGroupData[{
Cell[4253, 118, 566, 16, 85, "Input",ExpressionUUID->"0112db79-82b2-40a5-9583-c1b7738c4c4c"],
Cell[4822, 136, 370, 6, 55, "Output",ExpressionUUID->"d88d71cf-5fd8-45c1-95b1-2e73e9a087be"],
Cell[5195, 144, 370, 6, 55, "Output",ExpressionUUID->"38aa5c2a-e135-4b55-b9d4-484ad3e27afa"]
}, Open  ]]
}, Open  ]],
Cell[CellGroupData[{
Cell[5614, 156, 153, 3, 73, "Subsubsection",ExpressionUUID->"bde8c775-c83a-4510-b0ca-ff041ee2a122"],
Cell[5770, 161, 164, 3, 56, "Text",ExpressionUUID->"0fc09061-e01e-4f69-8870-8c827fc41dff"],
Cell[CellGroupData[{
Cell[5959, 168, 1112, 30, 85, "Input",ExpressionUUID->"3043dc14-1a85-4df4-b63a-8c8bf3dea910"],
Cell[7074, 200, 667, 19, 78, "Output",ExpressionUUID->"26e12a7b-159a-46a8-bfcc-eec29e629dc2"]
}, Open  ]],
Cell[7756, 222, 165, 3, 56, "Text",ExpressionUUID->"e63e1c12-a1fc-401c-a1b1-1fe3210d22c1"],
Cell[CellGroupData[{
Cell[7946, 229, 540, 15, 85, "Input",ExpressionUUID->"918474b6-3d1d-43e5-9c80-78d4b4b819a0"],
Cell[8489, 246, 293, 4, 55, "Output",ExpressionUUID->"68bb4015-4c4e-4dc0-80f6-3ea4eed4a223"],
Cell[8785, 252, 293, 4, 55, "Output",ExpressionUUID->"b21d99d6-c5bf-49e8-aded-c3f8281d8c78"]
}, Open  ]]
}, Open  ]],
Cell[CellGroupData[{
Cell[9127, 262, 155, 3, 73, "Subsubsection",ExpressionUUID->"422a493f-06cb-40a4-8f1e-03d08b4f607c"],
Cell[9285, 267, 140, 3, 56, "Text",ExpressionUUID->"2e2ed5b2-aeb4-4f08-a856-630ac1ca4c90"],
Cell[CellGroupData[{
Cell[9450, 274, 1006, 28, 85, "Input",ExpressionUUID->"2ad49d71-dd74-45b0-908d-0ce3169206f4"],
Cell[10459, 304, 668, 19, 78, "Output",ExpressionUUID->"90a1576c-6a0c-4184-b709-62f06a70712f"]
}, Open  ]],
Cell[11142, 326, 165, 3, 56, "Text",ExpressionUUID->"112ebb53-6b0b-4781-b847-ee3a1450365b"],
Cell[CellGroupData[{
Cell[11332, 333, 542, 15, 85, "Input",ExpressionUUID->"64f13582-4f3d-4b5a-a582-687056326408"],
Cell[11877, 350, 268, 4, 55, "Output",ExpressionUUID->"8a2a743e-75af-4c47-abbd-0885c7e6e7c7"],
Cell[12148, 356, 270, 4, 88, "Output",ExpressionUUID->"a1227b00-99f1-4653-b22d-a4fb81ae818e"]
}, Open  ]]
}, Open  ]]
}
]
*)

(* End of internal cache information *)

(* NotebookSignature bvD4cBd7EO0s7Dw7H1ocE5kg *)
