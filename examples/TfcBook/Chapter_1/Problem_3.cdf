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
NotebookDataLength[     12145,        338]
NotebookOptionsPosition[     11582,        318]
NotebookOutlinePosition[     11944,        334]
CellTagsIndexPosition[     11901,        331]
WindowFrame->Normal*)

(* Beginning of Notebook Content *)
Notebook[{

Cell[CellGroupData[{
Cell["Chapter 1 Problem 3:", "Subsubsection",
 CellChangeTimes->{{3.820586254287455*^9, 3.820586272280244*^9}, {
  3.8205879464678373`*^9, 3.820587946565063*^9}, {3.820592053273141*^9, 
  3.820592053356666*^9}},ExpressionUUID->"699b74c8-e698-404e-8ffd-\
7bdebbfbe347"],

Cell["Constrained expression:", "Text",
 CellChangeTimes->{{3.82058627523597*^9, 3.820586277701434*^9}, {
   3.820586345940712*^9, 3.820586347755028*^9}, {3.8205864717813807`*^9, 
   3.820586496739429*^9}, 
   3.8250943939809637`*^9},ExpressionUUID->"bb869408-732e-44bb-9089-\
813d400fe162"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"y", "[", "x_", "]"}], ":=", 
   RowBox[{
    RowBox[{"g", "[", "x", "]"}], "+", 
    RowBox[{
     FractionBox["1", "3"], " ", 
     RowBox[{"(", 
      RowBox[{
       RowBox[{"-", "3"}], "+", "x"}], ")"}], " ", 
     RowBox[{"(", 
      RowBox[{
       RowBox[{"-", "1"}], "+", "x"}], ")"}], 
     RowBox[{"(", 
      RowBox[{
       RowBox[{"g", "[", "1", "]"}], "-", 
       RowBox[{"g", "[", "0", "]"}]}], ")"}]}], "+", 
    RowBox[{
     FractionBox["1", "6"], " ", 
     RowBox[{"(", 
      RowBox[{
       RowBox[{"-", "3"}], "+", 
       RowBox[{"2", " ", 
        RowBox[{"(", 
         RowBox[{
          RowBox[{"-", "1"}], "+", "x"}], ")"}], " ", "x"}]}], ")"}], 
     RowBox[{"(", 
      RowBox[{"\[Pi]", "-", 
       RowBox[{"(", 
        RowBox[{
         RowBox[{"D", "[", 
          RowBox[{
           RowBox[{"g", "[", "\[Tau]", "]"}], ",", "\[Tau]"}], "]"}], "/.", 
         RowBox[{"\[Tau]", "\[Rule]", "2"}]}], ")"}]}], ")"}]}], "+", 
    RowBox[{
     RowBox[{"2", "/", "3"}], "*", 
     RowBox[{"(", 
      RowBox[{
       RowBox[{"-", "1"}], "-", 
       RowBox[{
        RowBox[{"1", "/", "2"}], "*", 
        RowBox[{"Integrate", "[", 
         RowBox[{
          RowBox[{"g", "[", "\[Tau]", "]"}], ",", 
          RowBox[{"{", 
           RowBox[{"\[Tau]", ",", "0", ",", "3"}], "}"}]}], "]"}]}]}], 
      ")"}]}]}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"y", "[", "x", "]"}], "//", "TraditionalForm"}]}], "Input",
 CellChangeTimes->{{3.8205862854499063`*^9, 3.8205863410355873`*^9}, {
  3.820586458036991*^9, 3.820586460723352*^9}, {3.820587569073257*^9, 
  3.820587591257009*^9}, {3.820587723983576*^9, 3.820587744850932*^9}, {
  3.820592096952223*^9, 3.82059209992798*^9}, {3.820592290632847*^9, 
  3.820592369492016*^9}, {3.820592453212667*^9, 3.820592463652636*^9}, {
  3.820592982045623*^9, 3.8205929822848873`*^9}, {3.8250944057637577`*^9, 
  3.825094418413087*^9}, {3.825094458215904*^9, 3.825094468385509*^9}},
 CellLabel->"In[1]:=",ExpressionUUID->"4356ffef-aa16-4ed7-a16e-04cbef825621"],

Cell[BoxData[
 FormBox[
  RowBox[{
   RowBox[{
    FractionBox["1", "6"], " ", 
    RowBox[{"(", 
     RowBox[{
      RowBox[{"2", " ", 
       RowBox[{"(", 
        RowBox[{"x", "-", "1"}], ")"}], " ", "x"}], "-", "3"}], ")"}], " ", 
    RowBox[{"(", 
     RowBox[{"\[Pi]", "-", 
      RowBox[{
       SuperscriptBox["g", "\[Prime]",
        MultilineFunction->None], "(", "2", ")"}]}], ")"}]}], "+", 
   RowBox[{
    FractionBox["2", "3"], " ", 
    RowBox[{"(", 
     RowBox[{
      RowBox[{"-", 
       RowBox[{
        FractionBox["1", "2"], " ", 
        RowBox[{
         SubsuperscriptBox["\[Integral]", "0", "3"], 
         RowBox[{
          RowBox[{"g", "(", "\[Tau]", ")"}], 
          RowBox[{"\[DifferentialD]", "\[Tau]"}]}]}]}]}], "-", "1"}], ")"}]}],
    "+", 
   RowBox[{
    FractionBox["1", "3"], " ", 
    RowBox[{"(", 
     RowBox[{
      RowBox[{"g", "(", "1", ")"}], "-", 
      RowBox[{"g", "(", "0", ")"}]}], ")"}], " ", 
    RowBox[{"(", 
     RowBox[{"x", "-", "3"}], ")"}], " ", 
    RowBox[{"(", 
     RowBox[{"x", "-", "1"}], ")"}]}], "+", 
   RowBox[{"g", "(", "x", ")"}]}], TraditionalForm]], "Output",
 CellChangeTimes->{{3.825094460159482*^9, 3.8250944694506283`*^9}, 
   3.825094503578125*^9, 3.829829952496463*^9},
 CellLabel->
  "Out[2]//TraditionalForm=",ExpressionUUID->"6787e46e-d46a-4413-a0e2-\
4362e83f108c"]
}, Open  ]],

Cell["\<\
Check the constraints (note last is true after simplification):\
\>", "Text",
 CellChangeTimes->{{3.820586351355624*^9, 3.820586356156693*^9}, 
   3.825094391893084*^9, {3.825094485050435*^9, 
   3.825094489418425*^9}},ExpressionUUID->"d889e916-a4db-4c04-9f2d-\
93d823b43940"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{
    RowBox[{"y", "[", "1", "]"}], "-", 
    RowBox[{"y", "[", "0", "]"}]}], " ", "\[Equal]", " ", "0"}], 
  "]"}], "\[IndentingNewLine]", 
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{
    RowBox[{"(", 
     RowBox[{
      RowBox[{"D", "[", 
       RowBox[{
        RowBox[{"y", "[", "x", "]"}], ",", "x"}], "]"}], "/.", 
      RowBox[{"x", "\[Rule]", "2"}]}], ")"}], "-", "\[Pi]"}], "\[Equal]", " ",
    "0"}], "]"}], "\[IndentingNewLine]", 
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{
    RowBox[{
     RowBox[{"1", "/", "2"}], "*", 
     RowBox[{"Integrate", "[", 
      RowBox[{
       RowBox[{"y", "[", "x", "]"}], ",", 
       RowBox[{"{", 
        RowBox[{"x", ",", "0", ",", "3"}], "}"}]}], "]"}]}], "+", "1"}], 
   "\[Equal]", " ", "0"}], "]"}]}], "Input",
 CellChangeTimes->{{3.820586368649537*^9, 3.820586402336206*^9}, 
   3.820586534390914*^9, {3.8205923713620853`*^9, 3.820592439991108*^9}, {
   3.820592628865643*^9, 3.820592644276186*^9}, {3.8205929852264757`*^9, 
   3.820592985405315*^9}},
 CellLabel->"In[3]:=",ExpressionUUID->"0112db79-82b2-40a5-9583-c1b7738c4c4c"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.820586396556509*^9, 3.820586429999325*^9, 3.8205864632424583`*^9, 
   3.820587137977333*^9, {3.820587573394577*^9, 3.820587593834496*^9}, 
   3.820587748196183*^9, 3.8205879522805853`*^9, {3.8205924403213987`*^9, 
   3.8205924661949883`*^9}, {3.820592630434806*^9, 3.8205926448890057`*^9}, {
   3.82059298578858*^9, 3.820593011142578*^9}, 3.8250945036817217`*^9, 
   3.829829952675343*^9},
 CellLabel->"Out[3]=",ExpressionUUID->"e3c905f5-61c0-479e-9b2f-a7502390d988"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.820586396556509*^9, 3.820586429999325*^9, 3.8205864632424583`*^9, 
   3.820587137977333*^9, {3.820587573394577*^9, 3.820587593834496*^9}, 
   3.820587748196183*^9, 3.8205879522805853`*^9, {3.8205924403213987`*^9, 
   3.8205924661949883`*^9}, {3.820592630434806*^9, 3.8205926448890057`*^9}, {
   3.82059298578858*^9, 3.820593011142578*^9}, 3.8250945036817217`*^9, 
   3.829829952689522*^9},
 CellLabel->"Out[4]=",ExpressionUUID->"1d96f095-1d1d-4c29-bcf2-2d8e6b54a4db"],

Cell[BoxData[
 RowBox[{
  RowBox[{"2", "+", 
   RowBox[{
    SubsuperscriptBox["\[Integral]", "0", "3"], 
    RowBox[{
     RowBox[{
      FractionBox["1", "6"], " ", 
      RowBox[{"(", 
       RowBox[{
        RowBox[{"2", " ", 
         RowBox[{"(", 
          RowBox[{
           RowBox[{"-", "3"}], "+", "x"}], ")"}], " ", 
         RowBox[{"(", 
          RowBox[{
           RowBox[{"-", "1"}], "+", "x"}], ")"}], " ", 
         RowBox[{"(", 
          RowBox[{
           RowBox[{"-", 
            RowBox[{"g", "[", "0", "]"}]}], "+", 
           RowBox[{"g", "[", "1", "]"}]}], ")"}]}], "+", 
        RowBox[{"6", " ", 
         RowBox[{"g", "[", "x", "]"}]}], "-", 
        RowBox[{"2", " ", 
         RowBox[{"(", 
          RowBox[{"2", "+", 
           RowBox[{
            SubsuperscriptBox["\[Integral]", "0", "3"], 
            RowBox[{
             RowBox[{"g", "[", "\[Tau]", "]"}], 
             RowBox[{"\[DifferentialD]", "\[Tau]"}]}]}]}], ")"}]}], "+", 
        RowBox[{
         RowBox[{"(", 
          RowBox[{
           RowBox[{"-", "3"}], "+", 
           RowBox[{"2", " ", 
            RowBox[{"(", 
             RowBox[{
              RowBox[{"-", "1"}], "+", "x"}], ")"}], " ", "x"}]}], ")"}], " ", 
         RowBox[{"(", 
          RowBox[{"\[Pi]", "-", 
           RowBox[{
            SuperscriptBox["g", "\[Prime]",
             MultilineFunction->None], "[", "2", "]"}]}], ")"}]}]}], ")"}]}], 
     
     RowBox[{"\[DifferentialD]", "x"}]}]}]}], "\[Equal]", "0"}]], "Output",
 CellChangeTimes->{
  3.820586396556509*^9, 3.820586429999325*^9, 3.8205864632424583`*^9, 
   3.820587137977333*^9, {3.820587573394577*^9, 3.820587593834496*^9}, 
   3.820587748196183*^9, 3.8205879522805853`*^9, {3.8205924403213987`*^9, 
   3.8205924661949883`*^9}, {3.820592630434806*^9, 3.8205926448890057`*^9}, {
   3.82059298578858*^9, 3.820593011142578*^9}, 3.8250945036817217`*^9, 
   3.8298299533085613`*^9},
 CellLabel->"Out[5]=",ExpressionUUID->"ce785a45-a45f-4637-91aa-fe06a74dbe2c"]
}, Open  ]],

Cell["Check integral constraint with some g:", "Text",
 CellChangeTimes->{{3.8205928029535723`*^9, 
  3.820592806815692*^9}},ExpressionUUID->"d88868af-06a6-440e-8330-\
2c17204d8ecb"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"g", "[", "x_", "]"}], ":=", 
   RowBox[{
    SuperscriptBox["x", "2"], " ", 
    RowBox[{"Sin", "[", "x", "]"}]}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{"N", "[", 
    RowBox[{"FullSimplify", "[", 
     RowBox[{
      RowBox[{
       RowBox[{"1", "/", "2"}], "*", 
       RowBox[{"Integrate", "[", 
        RowBox[{
         RowBox[{"y", "[", "x", "]"}], ",", 
         RowBox[{"{", 
          RowBox[{"x", ",", "0", ",", "3"}], "}"}]}], "]"}]}], "+", "1"}], 
     "]"}], "]"}], "\[Equal]", "0"}], "]"}], "\[IndentingNewLine]"}], "Input",\

 CellChangeTimes->{{3.82059280974417*^9, 3.820592861535864*^9}, {
  3.820592948870829*^9, 3.82059296782096*^9}, {3.8205930612287416`*^9, 
  3.820593133099875*^9}},
 CellLabel->"In[6]:=",ExpressionUUID->"cf6e7ce4-265f-4d13-a7fd-e62d1a899db5"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{{3.8205928255291977`*^9, 3.8205928618995934`*^9}, {
   3.8205929514720097`*^9, 3.820592968698369*^9}, 3.820593012448592*^9, {
   3.820593092326211*^9, 3.8205931337620583`*^9}, 3.8250945047280207`*^9, 
   3.829829953768608*^9},
 CellLabel->"Out[7]=",ExpressionUUID->"949136a5-d65b-45d7-82b6-5a9e1aad4884"]
}, Open  ]]
}, Open  ]]
},
WindowSize->{960, 704},
WindowMargins->{{Automatic, 0}, {0, Automatic}},
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
Cell[CellGroupData[{
Cell[1510, 35, 268, 4, 73, "Subsubsection",ExpressionUUID->"699b74c8-e698-404e-8ffd-7bdebbfbe347"],
Cell[1781, 41, 291, 5, 56, "Text",ExpressionUUID->"bb869408-732e-44bb-9089-813d400fe162"],
Cell[CellGroupData[{
Cell[2097, 50, 2089, 57, 210, "Input",ExpressionUUID->"4356ffef-aa16-4ed7-a16e-04cbef825621"],
Cell[4189, 109, 1350, 43, 165, "Output",ExpressionUUID->"6787e46e-d46a-4413-a0e2-4362e83f108c"]
}, Open  ]],
Cell[5554, 155, 286, 6, 56, "Text",ExpressionUUID->"d889e916-a4db-4c04-9f2d-93d823b43940"],
Cell[CellGroupData[{
Cell[5865, 165, 1172, 32, 120, "Input",ExpressionUUID->"0112db79-82b2-40a5-9583-c1b7738c4c4c"],
Cell[7040, 199, 523, 8, 55, "Output",ExpressionUUID->"e3c905f5-61c0-479e-9b2f-a7502390d988"],
Cell[7566, 209, 523, 8, 55, "Output",ExpressionUUID->"1d96f095-1d1d-4c29-bcf2-2d8e6b54a4db"],
Cell[8092, 219, 2004, 54, 149, "Output",ExpressionUUID->"ce785a45-a45f-4637-91aa-fe06a74dbe2c"]
}, Open  ]],
Cell[10111, 276, 182, 3, 56, "Text",ExpressionUUID->"d88868af-06a6-440e-8330-2c17204d8ecb"],
Cell[CellGroupData[{
Cell[10318, 283, 878, 24, 155, "Input",ExpressionUUID->"cf6e7ce4-265f-4d13-a7fd-e62d1a899db5"],
Cell[11199, 309, 355, 5, 55, "Output",ExpressionUUID->"949136a5-d65b-45d7-82b6-5a9e1aad4884"]
}, Open  ]]
}, Open  ]]
}
]
*)

(* End of internal cache information *)

(* NotebookSignature kxpgbV4ihyVITA18YxIIXN@# *)
