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
NotebookDataLength[     11925,        320]
NotebookOptionsPosition[     11240,        297]
NotebookOutlinePosition[     11603,        313]
CellTagsIndexPosition[     11560,        310]
WindowFrame->Normal*)

(* Beginning of Notebook Content *)
Notebook[{

Cell[CellGroupData[{
Cell["Chapter 1 Problem 4:", "Subsubsection",
 CellChangeTimes->{{3.820586254287455*^9, 3.820586272280244*^9}, {
  3.8205879464678373`*^9, 3.820587946565063*^9}, {3.820592053273141*^9, 
  3.820592053356666*^9}, {3.820593525363361*^9, 
  3.820593525384062*^9}},ExpressionUUID->"699b74c8-e698-404e-8ffd-\
7bdebbfbe347"],

Cell["Constrained expression", "Text",
 CellChangeTimes->{{3.82058627523597*^9, 3.820586277701434*^9}, {
  3.820586345940712*^9, 3.820586347755028*^9}, {3.8205864717813807`*^9, 
  3.820586496739429*^9}},ExpressionUUID->"bb869408-732e-44bb-9089-\
813d400fe162"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"yhat", "[", "x_", "]"}], ":=", 
   RowBox[{
    RowBox[{"g", "[", "x", "]"}], "+", 
    RowBox[{
     RowBox[{
      RowBox[{"(", 
       RowBox[{"3", "-", "x"}], ")"}], "/", "3"}], " ", 
     RowBox[{"(", 
      RowBox[{"1", "-", 
       RowBox[{"g", "[", "0", "]"}]}], ")"}]}], "+", 
    RowBox[{
     RowBox[{"x", "/", "3"}], 
     RowBox[{"(", 
      RowBox[{"\[Pi]", "-", 
       RowBox[{"g", "[", "3", "]"}]}], ")"}]}]}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"y", "[", "x_", "]"}], ":=", 
   RowBox[{
    RowBox[{"yhat", "[", "x", "]"}], "+", 
    RowBox[{
     RowBox[{"(", 
      RowBox[{
       RowBox[{"Sin", "[", "x", "]"}], "-", 
       RowBox[{"yhat", "[", "x", "]"}]}], ")"}], 
     RowBox[{"UnitStep", "[", 
      RowBox[{
       RowBox[{"Sin", "[", "x", "]"}], "-", 
       RowBox[{"yhat", "[", "x", "]"}]}], "]"}]}], "+", 
    RowBox[{
     RowBox[{"(", 
      RowBox[{
       RowBox[{"x", " ", 
        RowBox[{"Sqrt", "[", "x", "]"}]}], "+", "2", "-", 
       RowBox[{"yhat", "[", "x", "]"}]}], ")"}], 
     RowBox[{"UnitStep", "[", 
      RowBox[{
       RowBox[{"yhat", "[", "x", "]"}], "-", 
       RowBox[{"x", " ", 
        RowBox[{"Sqrt", "[", "x", "]"}]}], "-", "2"}], "]"}]}]}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"y", "[", "x", "]"}], "//", "TraditionalForm"}]}], "Input",
 CellChangeTimes->{{3.8205862854499063`*^9, 3.8205863410355873`*^9}, {
  3.820586458036991*^9, 3.820586460723352*^9}, {3.820587569073257*^9, 
  3.820587591257009*^9}, {3.820587723983576*^9, 3.820587744850932*^9}, {
  3.820592096952223*^9, 3.82059209992798*^9}, {3.820592290632847*^9, 
  3.820592369492016*^9}, {3.820592453212667*^9, 3.820592463652636*^9}, {
  3.820592982045623*^9, 3.8205929822848873`*^9}, {3.8205935296603603`*^9, 
  3.820593671970043*^9}, {3.820593814354515*^9, 3.8205938145760612`*^9}, {
  3.8250948064476833`*^9, 3.825094811055107*^9}},
 CellLabel->"In[1]:=",ExpressionUUID->"4356ffef-aa16-4ed7-a16e-04cbef825621"],

Cell[BoxData[
 FormBox[
  RowBox[{
   RowBox[{
    RowBox[{"(", 
     RowBox[{
      RowBox[{"-", 
       RowBox[{
        FractionBox["1", "3"], " ", 
        RowBox[{"(", 
         RowBox[{"\[Pi]", "-", 
          RowBox[{"g", "(", "3", ")"}]}], ")"}], " ", "x"}]}], "-", 
      RowBox[{
       FractionBox["1", "3"], " ", 
       RowBox[{"(", 
        RowBox[{"1", "-", 
         RowBox[{"g", "(", "0", ")"}]}], ")"}], " ", 
       RowBox[{"(", 
        RowBox[{"3", "-", "x"}], ")"}]}], "-", 
      RowBox[{"g", "(", "x", ")"}], "+", 
      SuperscriptBox["x", 
       RowBox[{"3", "/", "2"}]], "+", "2"}], ")"}], " ", 
    TemplateBox[{RowBox[{
        RowBox[{
          RowBox[{"-", 
            SuperscriptBox["x", 
             RowBox[{"3", "/", "2"}]]}], "+", 
          RowBox[{
            FractionBox["1", "3"], " ", 
            RowBox[{"(", 
              RowBox[{"\[Pi]", "-", 
                RowBox[{"g", "(", "3", ")"}]}], ")"}], " ", "x"}], "+", 
          RowBox[{
            FractionBox["1", "3"], " ", 
            RowBox[{"(", 
              RowBox[{"3", "-", "x"}], ")"}], " ", 
            RowBox[{"(", 
              RowBox[{"1", "-", 
                RowBox[{"g", "(", "0", ")"}]}], ")"}]}], "+", 
          RowBox[{"g", "(", "x", ")"}], "-", "2"}]}]},
     "UnitStepSeq"]}], "+", 
   RowBox[{
    RowBox[{"(", 
     RowBox[{
      RowBox[{"-", 
       RowBox[{
        FractionBox["1", "3"], " ", 
        RowBox[{"(", 
         RowBox[{"1", "-", 
          RowBox[{"g", "(", "0", ")"}]}], ")"}], " ", 
        RowBox[{"(", 
         RowBox[{"3", "-", "x"}], ")"}]}]}], "-", 
      RowBox[{
       FractionBox["1", "3"], " ", 
       RowBox[{"(", 
        RowBox[{"\[Pi]", "-", 
         RowBox[{"g", "(", "3", ")"}]}], ")"}], " ", "x"}], "-", 
      RowBox[{"g", "(", "x", ")"}], "+", 
      RowBox[{"sin", "(", "x", ")"}]}], ")"}], " ", 
    TemplateBox[{RowBox[{
        RowBox[{
          RowBox[{
            RowBox[{"-", 
              FractionBox["1", "3"]}], " ", 
            RowBox[{"(", 
              RowBox[{"3", "-", "x"}], ")"}], " ", 
            RowBox[{"(", 
              RowBox[{"1", "-", 
                RowBox[{"g", "(", "0", ")"}]}], ")"}]}], "-", 
          RowBox[{
            FractionBox["1", "3"], " ", "x", " ", 
            RowBox[{"(", 
              RowBox[{"\[Pi]", "-", 
                RowBox[{"g", "(", "3", ")"}]}], ")"}]}], "-", 
          RowBox[{"g", "(", "x", ")"}], "+", 
          RowBox[{"sin", "(", "x", ")"}]}]}]},
     "UnitStepSeq"]}], "+", 
   RowBox[{
    FractionBox["1", "3"], " ", 
    RowBox[{"(", 
     RowBox[{"1", "-", 
      RowBox[{"g", "(", "0", ")"}]}], ")"}], " ", 
    RowBox[{"(", 
     RowBox[{"3", "-", "x"}], ")"}]}], "+", 
   RowBox[{
    FractionBox["1", "3"], " ", 
    RowBox[{"(", 
     RowBox[{"\[Pi]", "-", 
      RowBox[{"g", "(", "3", ")"}]}], ")"}], " ", "x"}], "+", 
   RowBox[{"g", "(", "x", ")"}]}], TraditionalForm]], "Output",
 CellChangeTimes->{3.825094820140671*^9, 3.8298299817787933`*^9},
 CellLabel->
  "Out[3]//TraditionalForm=",ExpressionUUID->"073e95f9-a595-4af1-841c-\
7b2a73c5fe6e"]
}, Open  ]],

Cell["Check the value-level equality constraints", "Text",
 CellChangeTimes->{{3.820586351355624*^9, 3.820586356156693*^9}, {
  3.820593852451447*^9, 
  3.820593859559873*^9}},ExpressionUUID->"d889e916-a4db-4c04-9f2d-\
93d823b43940"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{
    RowBox[{"y", "[", "0", "]"}], "-", "1"}], " ", "\[Equal]", " ", "0"}], 
  "]"}], "\[IndentingNewLine]", 
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{"\[Pi]", "-", 
    RowBox[{"y", "[", "3", "]"}]}], "\[Equal]", " ", "0"}], "]"}]}], "Input",
 CellChangeTimes->{{3.820586368649537*^9, 3.820586402336206*^9}, 
   3.820586534390914*^9, {3.8205923713620853`*^9, 3.820592439991108*^9}, {
   3.820592628865643*^9, 3.820592644276186*^9}, {3.8205929852264757`*^9, 
   3.820592985405315*^9}, {3.82059367727635*^9, 3.820593687956897*^9}, {
   3.820593734073501*^9, 3.8205937343907127`*^9}, {3.820593787659169*^9, 
   3.820593790595353*^9}, {3.820593821049561*^9, 3.820593849742961*^9}},
 CellLabel->"In[4]:=",ExpressionUUID->"0112db79-82b2-40a5-9583-c1b7738c4c4c"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.820586396556509*^9, 3.820586429999325*^9, 3.8205864632424583`*^9, 
   3.820587137977333*^9, {3.820587573394577*^9, 3.820587593834496*^9}, 
   3.820587748196183*^9, 3.8205879522805853`*^9, {3.8205924403213987`*^9, 
   3.8205924661949883`*^9}, {3.820592630434806*^9, 3.8205926448890057`*^9}, {
   3.82059298578858*^9, 3.820593011142578*^9}, {3.820593693135427*^9, 
   3.82059373949123*^9}, {3.820593788034072*^9, 3.8205938669267473`*^9}, 
   3.820593929905636*^9, 3.820594135047867*^9, 3.825094820238823*^9, 
   3.829829981935206*^9},
 CellLabel->"Out[4]=",ExpressionUUID->"35438a1b-d875-4b4a-840d-a1e8950608ac"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.820586396556509*^9, 3.820586429999325*^9, 3.8205864632424583`*^9, 
   3.820587137977333*^9, {3.820587573394577*^9, 3.820587593834496*^9}, 
   3.820587748196183*^9, 3.8205879522805853`*^9, {3.8205924403213987`*^9, 
   3.8205924661949883`*^9}, {3.820592630434806*^9, 3.8205926448890057`*^9}, {
   3.82059298578858*^9, 3.820593011142578*^9}, {3.820593693135427*^9, 
   3.82059373949123*^9}, {3.820593788034072*^9, 3.8205938669267473`*^9}, 
   3.820593929905636*^9, 3.820594135047867*^9, 3.825094820238823*^9, 
   3.8298299819364567`*^9},
 CellLabel->"Out[5]=",ExpressionUUID->"3368da81-3104-416f-acff-cd5e070dd30e"]
}, Open  ]],

Cell["Check the inequality constraints", "Text",
 CellChangeTimes->{{3.820593882228283*^9, 
  3.820593887205159*^9}},ExpressionUUID->"123f5dc0-9723-4977-b635-\
43a7ee616fc0"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{"g", "[", "x_", "]"}], ":=", " ", "100"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"y", "[", "2", "]"}], " ", "\[LessEqual]", " ", 
  RowBox[{
   RowBox[{"2", " ", 
    RowBox[{"Sqrt", "[", "2", "]"}]}], "+", "2"}]}]}], "Input",
 CellChangeTimes->{{3.820594080199547*^9, 3.820594099745948*^9}},
 CellLabel->"In[6]:=",ExpressionUUID->"fd875e20-fa75-492e-a82e-ec8e76ad1f82"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{3.8205941008719482`*^9, 3.820594135107792*^9, 
  3.8250948202740107`*^9, 3.8298299819727182`*^9},
 CellLabel->"Out[7]=",ExpressionUUID->"799abb9f-55c3-4df2-9a2a-ca65b36270bb"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{"g", "[", "x_", "]"}], ":=", " ", 
  RowBox[{"-", "100"}]}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"y", "[", "2", "]"}], " ", "\[GreaterEqual]", "  ", 
  RowBox[{"Sin", "[", "2", "]"}]}]}], "Input",
 CellChangeTimes->{{3.820594108299737*^9, 3.820594120169874*^9}},
 CellLabel->"In[8]:=",ExpressionUUID->"732e7be3-fca7-42dc-ad2b-59d159e1b8d6"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{{3.820594120915846*^9, 3.820594135149921*^9}, 
   3.8250948203112707`*^9, 3.8298299820035753`*^9},
 CellLabel->"Out[9]=",ExpressionUUID->"e6f88e77-ac24-41b9-b8ef-be672d7a3f9a"]
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
Cell[CellGroupData[{
Cell[1510, 35, 317, 5, 73, "Subsubsection",ExpressionUUID->"699b74c8-e698-404e-8ffd-7bdebbfbe347"],
Cell[1830, 42, 260, 4, 56, "Text",ExpressionUUID->"bb869408-732e-44bb-9089-813d400fe162"],
Cell[CellGroupData[{
Cell[2115, 50, 2036, 55, 155, "Input",ExpressionUUID->"4356ffef-aa16-4ed7-a16e-04cbef825621"],
Cell[4154, 107, 3105, 93, 222, "Output",ExpressionUUID->"073e95f9-a595-4af1-841c-7b2a73c5fe6e"]
}, Open  ]],
Cell[7274, 203, 233, 4, 56, "Text",ExpressionUUID->"d889e916-a4db-4c04-9f2d-93d823b43940"],
Cell[CellGroupData[{
Cell[7532, 211, 836, 16, 85, "Input",ExpressionUUID->"0112db79-82b2-40a5-9583-c1b7738c4c4c"],
Cell[8371, 229, 666, 10, 55, "Output",ExpressionUUID->"35438a1b-d875-4b4a-840d-a1e8950608ac"],
Cell[9040, 241, 668, 10, 55, "Output",ExpressionUUID->"3368da81-3104-416f-acff-cd5e070dd30e"]
}, Open  ]],
Cell[9723, 254, 174, 3, 56, "Text",ExpressionUUID->"123f5dc0-9723-4977-b635-43a7ee616fc0"],
Cell[CellGroupData[{
Cell[9922, 261, 411, 9, 85, "Input",ExpressionUUID->"fd875e20-fa75-492e-a82e-ec8e76ad1f82"],
Cell[10336, 272, 226, 3, 55, "Output",ExpressionUUID->"799abb9f-55c3-4df2-9a2a-ca65b36270bb"]
}, Open  ]],
Cell[CellGroupData[{
Cell[10599, 280, 383, 8, 85, "Input",ExpressionUUID->"732e7be3-fca7-42dc-ad2b-59d159e1b8d6"],
Cell[10985, 290, 227, 3, 55, "Output",ExpressionUUID->"e6f88e77-ac24-41b9-b8ef-be672d7a3f9a"]
}, Open  ]]
}, Open  ]]
}
]
*)

(* End of internal cache information *)

(* NotebookSignature Ix05bD7zPrmBIDgf@Nu3L#17 *)
