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
NotebookDataLength[     21468,        630]
NotebookOptionsPosition[     18453,        571]
NotebookOutlinePosition[     18816,        587]
CellTagsIndexPosition[     18773,        584]
WindowFrame->Normal*)

(* Beginning of Notebook Content *)
Notebook[{
Cell["Chapter 1 Problem 1:", "Subsubsection",
 CellChangeTimes->{{3.820586254287455*^9, 
  3.820586272280244*^9}},ExpressionUUID->"699b74c8-e698-404e-8ffd-\
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
      RowBox[{"x", "-", "x0"}], ")"}], 
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
  3.820586458036991*^9, 3.820586460723352*^9}, {3.825094660181346*^9, 
  3.825094668556292*^9}},
 CellLabel->"In[1]:=",ExpressionUUID->"4356ffef-aa16-4ed7-a16e-04cbef825621"],

Cell[BoxData[
 FormBox[
  RowBox[{
   RowBox[{
    RowBox[{"(", 
     RowBox[{"x", "-", "x0"}], ")"}], " ", 
    RowBox[{"(", 
     RowBox[{"yx0", "-", 
      RowBox[{
       SuperscriptBox["g", "\[Prime]",
        MultilineFunction->None], "(", "x0", ")"}]}], ")"}]}], "+", 
   RowBox[{"g", "(", "x", ")"}], "-", 
   RowBox[{"g", "(", "x0", ")"}], "+", "y0"}], TraditionalForm]], "Output",
 CellChangeTimes->{3.825094669504422*^9, 3.825094712161126*^9, 
  3.829829900240033*^9},
 CellLabel->
  "Out[2]//TraditionalForm=",ExpressionUUID->"41bc590d-d5f3-4819-83ac-\
f5cefa4e48ab"]
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
 CellChangeTimes->{3.820586396556509*^9, 3.820586429999325*^9, 
  3.8205864632424583`*^9, 3.820587137977333*^9, 3.825094712215166*^9, 
  3.829829900394734*^9},
 CellLabel->"Out[3]=",ExpressionUUID->"6a0b0b44-e25b-48d1-9a9c-0fc1475dd114"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{3.820586396556509*^9, 3.820586429999325*^9, 
  3.8205864632424583`*^9, 3.820587137977333*^9, 3.825094712215166*^9, 
  3.829829900396452*^9},
 CellLabel->"Out[4]=",ExpressionUUID->"c7179812-ad64-4b5c-a6fc-e99bf04bc7ae"]
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
     RowBox[{"(", 
      RowBox[{"x", "-", "xf"}], ")"}], 
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
  3.825094682852303*^9, 3.8250946879955606`*^9}},
 CellLabel->"In[5]:=",ExpressionUUID->"3043dc14-1a85-4df4-b63a-8c8bf3dea910"],

Cell[BoxData[
 FormBox[
  RowBox[{
   RowBox[{
    RowBox[{"(", 
     RowBox[{"x", "-", "xf"}], ")"}], " ", 
    RowBox[{"(", 
     RowBox[{"yx0", "-", 
      RowBox[{
       SuperscriptBox["g", "\[Prime]",
        MultilineFunction->None], "(", "x0", ")"}]}], ")"}]}], "+", 
   RowBox[{"g", "(", "x", ")"}], "-", 
   RowBox[{"g", "(", "xf", ")"}], "+", "yf"}], TraditionalForm]], "Output",
 CellChangeTimes->{{3.82509468891391*^9, 3.825094712232153*^9}, 
   3.829829900434066*^9},
 CellLabel->
  "Out[6]//TraditionalForm=",ExpressionUUID->"023089e5-a538-4ad4-a4d5-\
fb209276d2be"]
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
  3.825094712262546*^9, 3.829829900472233*^9},
 CellLabel->"Out[7]=",ExpressionUUID->"e304ddce-76c2-45cc-9f14-c5871f3035bf"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{3.820586578018466*^9, 3.8205871380216227`*^9, 
  3.825094712262546*^9, 3.829829900475089*^9},
 CellLabel->"Out[8]=",ExpressionUUID->"43bcb95a-cb90-4d6e-b559-238ee0bf037c"]
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
     RowBox[{"(", 
      RowBox[{"x", "-", "x0"}], ")"}], 
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
 CellChangeTimes->{{3.820586594811557*^9, 3.820586644212625*^9}, {
  3.825094692723748*^9, 3.825094695699484*^9}},
 CellLabel->"In[9]:=",ExpressionUUID->"2ad49d71-dd74-45b0-908d-0ce3169206f4"],

Cell[BoxData[
 FormBox[
  RowBox[{
   RowBox[{
    RowBox[{"(", 
     RowBox[{"x", "-", "x0"}], ")"}], " ", 
    RowBox[{"(", 
     RowBox[{"yxf", "-", 
      RowBox[{
       SuperscriptBox["g", "\[Prime]",
        MultilineFunction->None], "(", "xf", ")"}]}], ")"}]}], "+", 
   RowBox[{"g", "(", "x", ")"}], "-", 
   RowBox[{"g", "(", "x0", ")"}], "+", "y0"}], TraditionalForm]], "Output",
 CellChangeTimes->{{3.825094696268713*^9, 3.8250947122898483`*^9}, 
   3.829829900508526*^9},
 CellLabel->
  "Out[10]//TraditionalForm=",ExpressionUUID->"2e27d90d-488d-44f9-aa98-\
c0a56da6a5f9"]
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
  3.8250947123156652`*^9, 3.8298299005421467`*^9},
 CellLabel->"Out[11]=",ExpressionUUID->"64c88e4e-56f2-451b-981f-7c332ebf6799"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{3.820586646760118*^9, 3.820587138064991*^9, 
  3.8250947123156652`*^9, 3.829829900544483*^9},
 CellLabel->"Out[12]=",ExpressionUUID->"e2f8f2c0-706b-43bb-b36d-8fc4779bf39c"]
}, Open  ]]
}, Open  ]],

Cell[CellGroupData[{

Cell["4.", "Subsubsection",
 CellChangeTimes->{{3.820586650755002*^9, 
  3.8205866509651213`*^9}},ExpressionUUID->"067657e5-9cb0-4319-9792-\
6fbc9ac0111d"],

Cell["Constrained expression", "Text",
 CellChangeTimes->{{3.820586668936166*^9, 
  3.820586670400752*^9}},ExpressionUUID->"36d13b0c-0eb4-4a3d-bf3b-\
f2ea375e709f"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"y4", "[", "x_", "]"}], ":=", 
   RowBox[{
    RowBox[{"g", "[", "x", "]"}], "+", 
    RowBox[{
     FractionBox[
      RowBox[{
       SuperscriptBox[
        RowBox[{"(", 
         RowBox[{"x", "-", "xf"}], ")"}], "2"], " ", 
       RowBox[{"(", 
        RowBox[{
         RowBox[{
          RowBox[{"-", "2"}], " ", "x"}], "+", 
         RowBox[{"3", " ", "x0"}], "-", "xf"}], ")"}]}], 
      SuperscriptBox[
       RowBox[{"(", 
        RowBox[{"x0", "-", "xf"}], ")"}], "3"]], 
     RowBox[{"(", 
      RowBox[{"y0", "-", 
       RowBox[{"g", "[", "x0", "]"}]}], ")"}]}], "+", 
    RowBox[{
     FractionBox[
      RowBox[{
       RowBox[{"(", 
        RowBox[{"x", "-", "x0"}], ")"}], " ", 
       SuperscriptBox[
        RowBox[{"(", 
         RowBox[{"x", "-", "xf"}], ")"}], "2"]}], 
      SuperscriptBox[
       RowBox[{"(", 
        RowBox[{"x0", "-", "xf"}], ")"}], "2"]], 
     RowBox[{"(", 
      RowBox[{"yx0", "-", 
       RowBox[{"(", 
        RowBox[{
         RowBox[{"D", "[", 
          RowBox[{
           RowBox[{"g", "[", "x", "]"}], ",", "x"}], "]"}], "/.", 
         RowBox[{"x", "\[Rule]", "x0"}]}], ")"}]}], ")"}]}], "+", 
    RowBox[{
     FractionBox[
      RowBox[{
       SuperscriptBox[
        RowBox[{"(", 
         RowBox[{"x", "-", "x0"}], ")"}], "2"], " ", 
       RowBox[{"(", 
        RowBox[{
         RowBox[{"2", " ", "x"}], "+", "x0", "-", 
         RowBox[{"3", " ", "xf"}]}], ")"}]}], 
      SuperscriptBox[
       RowBox[{"(", 
        RowBox[{"x0", "-", "xf"}], ")"}], "3"]], 
     RowBox[{"(", 
      RowBox[{"yf", "-", 
       RowBox[{"g", "[", "xf", "]"}]}], ")"}]}], "+", 
    RowBox[{
     FractionBox[
      RowBox[{
       SuperscriptBox[
        RowBox[{"(", 
         RowBox[{"x", "-", "x0"}], ")"}], "2"], " ", 
       RowBox[{"(", 
        RowBox[{"x", "-", "xf"}], ")"}]}], 
      SuperscriptBox[
       RowBox[{"(", 
        RowBox[{"x0", "-", "xf"}], ")"}], "2"]], 
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
  RowBox[{"y4", "[", "x", "]"}], "//", "TraditionalForm"}]}], "Input",
 CellChangeTimes->{{3.8205866757346478`*^9, 3.820586675818385*^9}, {
  3.8205868509038754`*^9, 3.820586905091144*^9}, {3.820587070056452*^9, 
  3.8205870827932997`*^9}, {3.825094700876231*^9, 3.8250947040030813`*^9}},
 CellLabel->"In[13]:=",ExpressionUUID->"81ab93e7-5512-47e7-a8a4-d8b8a8a8c611"],

Cell[BoxData[
 FormBox[
  RowBox[{
   FractionBox[
    RowBox[{
     RowBox[{"(", 
      RowBox[{"x", "-", "x0"}], ")"}], " ", 
     SuperscriptBox[
      RowBox[{"(", 
       RowBox[{"x", "-", "xf"}], ")"}], "2"], " ", 
     RowBox[{"(", 
      RowBox[{"yx0", "-", 
       RowBox[{
        SuperscriptBox["g", "\[Prime]",
         MultilineFunction->None], "(", "x0", ")"}]}], ")"}]}], 
    SuperscriptBox[
     RowBox[{"(", 
      RowBox[{"x0", "-", "xf"}], ")"}], "2"]], "+", 
   FractionBox[
    RowBox[{
     SuperscriptBox[
      RowBox[{"(", 
       RowBox[{"x", "-", "x0"}], ")"}], "2"], " ", 
     RowBox[{"(", 
      RowBox[{"x", "-", "xf"}], ")"}], " ", 
     RowBox[{"(", 
      RowBox[{"yxf", "-", 
       RowBox[{
        SuperscriptBox["g", "\[Prime]",
         MultilineFunction->None], "(", "xf", ")"}]}], ")"}]}], 
    SuperscriptBox[
     RowBox[{"(", 
      RowBox[{"x0", "-", "xf"}], ")"}], "2"]], "+", 
   FractionBox[
    RowBox[{
     SuperscriptBox[
      RowBox[{"(", 
       RowBox[{"x", "-", "xf"}], ")"}], "2"], " ", 
     RowBox[{"(", 
      RowBox[{"y0", "-", 
       RowBox[{"g", "(", "x0", ")"}]}], ")"}], " ", 
     RowBox[{"(", 
      RowBox[{
       RowBox[{"-", 
        RowBox[{"2", " ", "x"}]}], "+", 
       RowBox[{"3", " ", "x0"}], "-", "xf"}], ")"}]}], 
    SuperscriptBox[
     RowBox[{"(", 
      RowBox[{"x0", "-", "xf"}], ")"}], "3"]], "+", 
   FractionBox[
    RowBox[{
     SuperscriptBox[
      RowBox[{"(", 
       RowBox[{"x", "-", "x0"}], ")"}], "2"], " ", 
     RowBox[{"(", 
      RowBox[{"yf", "-", 
       RowBox[{"g", "(", "xf", ")"}]}], ")"}], " ", 
     RowBox[{"(", 
      RowBox[{
       RowBox[{"2", " ", "x"}], "+", "x0", "-", 
       RowBox[{"3", " ", "xf"}]}], ")"}]}], 
    SuperscriptBox[
     RowBox[{"(", 
      RowBox[{"x0", "-", "xf"}], ")"}], "3"]], "+", 
   RowBox[{"g", "(", "x", ")"}]}], TraditionalForm]], "Output",
 CellChangeTimes->{{3.825094704511713*^9, 3.825094712361388*^9}, 
   3.829829900594226*^9},
 CellLabel->
  "Out[14]//TraditionalForm=",ExpressionUUID->"fbedefdd-aa58-4914-9507-\
3ad3ce3fe850"]
}, Open  ]],

Cell["Check the constraints", "Text",
 CellChangeTimes->{{3.820586678826037*^9, 
  3.820586680138587*^9}},ExpressionUUID->"f00480fc-39de-463f-a437-\
c307b66d6035"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"y4", "[", "x0", "]"}], "-", "y0"}], " ", "\[Equal]", " ", 
  "0"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"(", 
    RowBox[{
     RowBox[{"D", "[", 
      RowBox[{
       RowBox[{"y4", "[", "x", "]"}], ",", "x"}], "]"}], "/.", 
     RowBox[{"x", "\[Rule]", "x0"}]}], ")"}], "-", "yx0"}], " ", "\[Equal]", 
  " ", "0"}], "\[IndentingNewLine]", 
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{
    RowBox[{"y4", "[", "xf", "]"}], "-", "yf"}], " ", "\[Equal]", " ", "0"}], 
  "]"}], "\[IndentingNewLine]", 
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{
    RowBox[{"(", 
     RowBox[{
      RowBox[{"D", "[", 
       RowBox[{
        RowBox[{"y4", "[", "x", "]"}], ",", "x"}], "]"}], "/.", 
      RowBox[{"x", "\[Rule]", "xf"}]}], ")"}], "-", "yxf"}], " ", "\[Equal]", 
   " ", "0"}], "]"}]}], "Input",
 CellChangeTimes->{{3.8205866845744543`*^9, 3.820586723560296*^9}, {
  3.820587095194772*^9, 3.820587109727358*^9}},
 CellLabel->"In[15]:=",ExpressionUUID->"2aad7361-4556-4ef7-815c-9f56d9d54fac"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.820586908639454*^9, {3.8205870852469*^9, 3.820587138151697*^9}, 
   3.825094712383977*^9, 3.8298299006321907`*^9},
 CellLabel->"Out[15]=",ExpressionUUID->"474e13cd-12d1-4815-98f1-427ecf7e9e5a"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.820586908639454*^9, {3.8205870852469*^9, 3.820587138151697*^9}, 
   3.825094712383977*^9, 3.829829900635375*^9},
 CellLabel->"Out[16]=",ExpressionUUID->"7daa3e13-68b3-41e2-90a7-07bf37b674ab"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.820586908639454*^9, {3.8205870852469*^9, 3.820587138151697*^9}, 
   3.825094712383977*^9, 3.829829900638249*^9},
 CellLabel->"Out[17]=",ExpressionUUID->"7c196ea9-ddbf-44c1-b002-ef6af7d5dc66"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.820586908639454*^9, {3.8205870852469*^9, 3.820587138151697*^9}, 
   3.825094712383977*^9, 3.829829900640996*^9},
 CellLabel->"Out[18]=",ExpressionUUID->"6963c76a-1d7b-48e9-9625-09c75af67cf0"]
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
Cell[1488, 33, 171, 3, 73, "Subsubsection",ExpressionUUID->"699b74c8-e698-404e-8ffd-7bdebbfbe347"],
Cell[CellGroupData[{
Cell[1684, 40, 155, 3, 58, "Subsubsection",ExpressionUUID->"625fcf49-195a-485a-89d1-d3f697221d08"],
Cell[1842, 45, 260, 4, 56, "Text",ExpressionUUID->"bb869408-732e-44bb-9089-813d400fe162"],
Cell[CellGroupData[{
Cell[2127, 53, 920, 26, 85, "Input",ExpressionUUID->"4356ffef-aa16-4ed7-a16e-04cbef825621"],
Cell[3050, 81, 579, 17, 78, "Output",ExpressionUUID->"41bc590d-d5f3-4819-83ac-f5cefa4e48ab"]
}, Open  ]],
Cell[3644, 101, 163, 3, 56, "Text",ExpressionUUID->"d889e916-a4db-4c04-9f2d-93d823b43940"],
Cell[CellGroupData[{
Cell[3832, 108, 566, 16, 85, "Input",ExpressionUUID->"0112db79-82b2-40a5-9583-c1b7738c4c4c"],
Cell[4401, 126, 269, 4, 55, "Output",ExpressionUUID->"6a0b0b44-e25b-48d1-9a9c-0fc1475dd114"],
Cell[4673, 132, 269, 4, 55, "Output",ExpressionUUID->"c7179812-ad64-4b5c-a6fc-e99bf04bc7ae"]
}, Open  ]]
}, Open  ]],
Cell[CellGroupData[{
Cell[4991, 142, 153, 3, 73, "Subsubsection",ExpressionUUID->"bde8c775-c83a-4510-b0ca-ff041ee2a122"],
Cell[5147, 147, 164, 3, 56, "Text",ExpressionUUID->"0fc09061-e01e-4f69-8870-8c827fc41dff"],
Cell[CellGroupData[{
Cell[5336, 154, 869, 25, 85, "Input",ExpressionUUID->"3043dc14-1a85-4df4-b63a-8c8bf3dea910"],
Cell[6208, 181, 581, 17, 78, "Output",ExpressionUUID->"023089e5-a538-4ad4-a4d5-fb209276d2be"]
}, Open  ]],
Cell[6804, 201, 165, 3, 56, "Text",ExpressionUUID->"e63e1c12-a1fc-401c-a1b1-1fe3210d22c1"],
Cell[CellGroupData[{
Cell[6994, 208, 540, 15, 85, "Input",ExpressionUUID->"918474b6-3d1d-43e5-9c80-78d4b4b819a0"],
Cell[7537, 225, 222, 3, 55, "Output",ExpressionUUID->"e304ddce-76c2-45cc-9f14-c5871f3035bf"],
Cell[7762, 230, 222, 3, 55, "Output",ExpressionUUID->"43bcb95a-cb90-4d6e-b559-238ee0bf037c"]
}, Open  ]]
}, Open  ]],
Cell[CellGroupData[{
Cell[8033, 239, 155, 3, 73, "Subsubsection",ExpressionUUID->"422a493f-06cb-40a4-8f1e-03d08b4f607c"],
Cell[8191, 244, 140, 3, 56, "Text",ExpressionUUID->"2e2ed5b2-aeb4-4f08-a856-630ac1ca4c90"],
Cell[CellGroupData[{
Cell[8356, 251, 867, 25, 85, "Input",ExpressionUUID->"2ad49d71-dd74-45b0-908d-0ce3169206f4"],
Cell[9226, 278, 585, 17, 78, "Output",ExpressionUUID->"2e27d90d-488d-44f9-aa98-c0a56da6a5f9"]
}, Open  ]],
Cell[9826, 298, 165, 3, 56, "Text",ExpressionUUID->"112ebb53-6b0b-4781-b847-ee3a1450365b"],
Cell[CellGroupData[{
Cell[10016, 305, 542, 15, 85, "Input",ExpressionUUID->"64f13582-4f3d-4b5a-a582-687056326408"],
Cell[10561, 322, 225, 3, 55, "Output",ExpressionUUID->"64c88e4e-56f2-451b-981f-7c332ebf6799"],
Cell[10789, 327, 223, 3, 55, "Output",ExpressionUUID->"e2f8f2c0-706b-43bb-b36d-8fc4779bf39c"]
}, Open  ]]
}, Open  ]],
Cell[CellGroupData[{
Cell[11061, 336, 155, 3, 73, "Subsubsection",ExpressionUUID->"067657e5-9cb0-4319-9792-6fbc9ac0111d"],
Cell[11219, 341, 164, 3, 56, "Text",ExpressionUUID->"36d13b0c-0eb4-4a3d-bf3b-f2ea375e709f"],
Cell[CellGroupData[{
Cell[11408, 348, 2649, 83, 278, "Input",ExpressionUUID->"81ab93e7-5512-47e7-a8a4-d8b8a8a8c611"],
Cell[14060, 433, 2085, 69, 183, "Output",ExpressionUUID->"fbedefdd-aa58-4914-9507-3ad3ce3fe850"]
}, Open  ]],
Cell[16160, 505, 163, 3, 56, "Text",ExpressionUUID->"f00480fc-39de-463f-a437-c307b66d6035"],
Cell[CellGroupData[{
Cell[16348, 512, 1075, 31, 155, "Input",ExpressionUUID->"2aad7361-4556-4ef7-815c-9f56d9d54fac"],
Cell[17426, 545, 249, 4, 55, "Output",ExpressionUUID->"474e13cd-12d1-4815-98f1-427ecf7e9e5a"],
Cell[17678, 551, 247, 4, 55, "Output",ExpressionUUID->"7daa3e13-68b3-41e2-90a7-07bf37b674ab"],
Cell[17928, 557, 247, 4, 55, "Output",ExpressionUUID->"7c196ea9-ddbf-44c1-b002-ef6af7d5dc66"],
Cell[18178, 563, 247, 4, 88, "Output",ExpressionUUID->"6963c76a-1d7b-48e9-9625-09c75af67cf0"]
}, Open  ]]
}, Open  ]]
}
]
*)

(* End of internal cache information *)

(* NotebookSignature TvTpWROWr8rzABgnY3mMtkOR *)
