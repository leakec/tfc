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
NotebookDataLength[     25572,        696]
NotebookOptionsPosition[     23713,        651]
NotebookOutlinePosition[     24076,        667]
CellTagsIndexPosition[     24033,        664]
WindowFrame->Normal*)

(* Beginning of Notebook Content *)
Notebook[{

Cell[CellGroupData[{
Cell["Create the constrained expression:", "Subsubsection",
 CellChangeTimes->{{3.827421030012843*^9, 
  3.8274210331015873`*^9}},ExpressionUUID->"c9b673bf-907a-4716-9647-\
816a981a0b8d"],

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"s1", "[", "x_", "]"}], ":=", 
   RowBox[{"{", "1", "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"S1", " ", "=", " ", 
   RowBox[{"{", 
    RowBox[{"Integrate", "[", 
     RowBox[{
      RowBox[{"Integrate", "[", 
       RowBox[{
        RowBox[{"s1", "[", "x", "]"}], ",", 
        RowBox[{"{", 
         RowBox[{"y", ",", "0", ",", "1"}], "}"}]}], "]"}], ",", 
      RowBox[{"{", 
       RowBox[{"z", ",", "0", ",", "2"}], "}"}]}], "]"}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"\[Alpha]1", " ", "=", " ", 
   RowBox[{"Inverse", "[", "S1", "]"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"\[Phi]1", "[", "x_", "]"}], ":=", 
   RowBox[{
    RowBox[{"s1", "[", "x", "]"}], ".", "\[Alpha]1"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"\[Rho]1", "[", 
    RowBox[{"x_", ",", "y_", ",", "z_", ",", "gu_Symbol"}], "]"}], ":=", 
   RowBox[{"{", 
    RowBox[{"1", "-", 
     RowBox[{"Integrate", "[", 
      RowBox[{
       RowBox[{"Integrate", "[", 
        RowBox[{
         RowBox[{"gu", "[", 
          RowBox[{"x", ",", "\[Alpha]", ",", "\[Beta]"}], "]"}], ",", 
         RowBox[{"{", 
          RowBox[{"\[Alpha]", ",", "0", ",", "1"}], "}"}]}], "]"}], ",", 
       RowBox[{"{", 
        RowBox[{"\[Beta]", ",", "0", ",", "2"}], "}"}]}], "]"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"u1", "[", 
     RowBox[{"x_", ",", "y_", ",", "z_", ",", "gu_Symbol"}], "]"}], ":=", 
    RowBox[{
     RowBox[{"gu", "[", 
      RowBox[{"x", ",", "y", ",", "z"}], "]"}], "+", 
     RowBox[{
      RowBox[{"\[Phi]1", "[", "x", "]"}], ".", 
      RowBox[{"\[Rho]1", "[", 
       RowBox[{"x", ",", "y", ",", "z", ",", "gu"}], "]"}]}]}]}], ";"}], 
  "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"s2", "[", "z_", "]"}], ":=", 
   RowBox[{"{", 
    RowBox[{"1", ",", "z"}], "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"S2", " ", "=", " ", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{
      RowBox[{"s2", "[", "0", "]"}], "-", 
      RowBox[{"s2", "[", "1", "]"}]}], ",", 
     RowBox[{"Integrate", "[", 
      RowBox[{
       RowBox[{"Integrate", "[", 
        RowBox[{
         RowBox[{"s2", "[", "z", "]"}], ",", 
         RowBox[{"{", 
          RowBox[{"y", ",", "0", ",", "1"}], "}"}]}], "]"}], ",", 
       RowBox[{"{", 
        RowBox[{"z", ",", "0", ",", "2"}], "}"}]}], "]"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"\[Alpha]2", " ", "=", " ", 
   RowBox[{
    RowBox[{"Inverse", "[", "S2", "]"}], ".", 
    RowBox[{"{", 
     RowBox[{
      RowBox[{"{", "1", "}"}], ",", 
      RowBox[{"{", "0", "}"}]}], "}"}]}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"\[Phi]2", "[", "z_", "]"}], ":=", 
   RowBox[{
    RowBox[{"s2", "[", "z", "]"}], ".", "\[Alpha]2"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"\[Rho]2", "[", 
    RowBox[{"x_", ",", "y_", ",", "z_", ",", "gu_Symbol"}], "]"}], ":=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"u1", "[", 
      RowBox[{"x", ",", "y", ",", "1", ",", "gu"}], "]"}], "-", 
     RowBox[{"u1", "[", 
      RowBox[{"x", ",", "y", ",", "0", ",", "gu"}], "]"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"u", "[", 
     RowBox[{"x_", ",", "y_", ",", "z_", ",", "gu_Symbol"}], "]"}], ":=", 
    RowBox[{
     RowBox[{"u1", "[", 
      RowBox[{"x", ",", "y", ",", "z", ",", "gu"}], "]"}], "+", 
     RowBox[{
      RowBox[{"\[Phi]2", "[", "z", "]"}], ".", 
      RowBox[{"\[Rho]2", "[", 
       RowBox[{"x", ",", "y", ",", "z", ",", "gu"}], "]"}]}]}]}], ";"}], 
  "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"s3", "[", "y_", "]"}], ":=", 
   RowBox[{"{", 
    RowBox[{"1", ",", "y"}], "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"S3", " ", "=", " ", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"s3", "[", "0", "]"}], ",", 
     RowBox[{
      RowBox[{"s3", "[", "1", "]"}], "+", 
      RowBox[{"(", 
       RowBox[{
        RowBox[{"D", "[", 
         RowBox[{
          RowBox[{"s3", "[", "y", "]"}], ",", "y"}], "]"}], "/.", 
        RowBox[{"y", "\[Rule]", "1"}]}], ")"}]}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"\[Alpha]3", " ", "=", " ", 
   RowBox[{"Inverse", "[", "S3", "]"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"\[Phi]3", "[", "y_", "]"}], ":=", 
   RowBox[{
    RowBox[{"s3", "[", "y", "]"}], ".", "\[Alpha]3"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"\[Rho]3", "[", 
    RowBox[{"x_", ",", "y_", ",", "z_", ",", "gu_Symbol", ",", "gv_Symbol"}], 
    "]"}], ":=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"5", "-", 
      RowBox[{"u", "[", 
       RowBox[{"x", ",", "0", ",", "z", ",", "gu"}], "]"}], "-", 
      RowBox[{"gv", "[", 
       RowBox[{"x", ",", "0", ",", "z"}], "]"}]}], ",", 
     RowBox[{"\[Pi]", "-", 
      RowBox[{"gv", "[", 
       RowBox[{"x", ",", "1", ",", "z"}], "]"}], "-", 
      RowBox[{"(", 
       RowBox[{
        RowBox[{"D", "[", 
         RowBox[{
          RowBox[{"gv", "[", 
           RowBox[{"x", ",", "\[Tau]", ",", "z"}], "]"}], ",", "\[Tau]"}], 
         "]"}], "/.", 
        RowBox[{"\[Tau]", "\[Rule]", "1"}]}], ")"}]}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"v", "[", 
    RowBox[{"x_", ",", "y_", ",", "z_", ",", "gu_Symbol", ",", "gv_Symbol"}], 
    "]"}], ":=", 
   RowBox[{
    RowBox[{"gv", "[", 
     RowBox[{"x", ",", "y", ",", "z"}], "]"}], "+", 
    RowBox[{
     RowBox[{"\[Phi]3", "[", "y", "]"}], ".", 
     RowBox[{"\[Rho]3", "[", 
      RowBox[{"x", ",", "y", ",", "z", ",", "gu", ",", "gv"}], "]"}]}]}]}], 
  ";"}]}], "Input",
 CellChangeTimes->{{3.82742108184127*^9, 3.8274211817179327`*^9}, {
   3.827421214877409*^9, 3.8274212464222383`*^9}, {3.827421358030734*^9, 
   3.827421712524737*^9}, {3.8274218368526297`*^9, 3.8274219484577093`*^9}, {
   3.827422040542109*^9, 3.8274221313900337`*^9}, {3.827422202033924*^9, 
   3.827422213349605*^9}, {3.827422469339161*^9, 3.827422469442131*^9}, {
   3.827423105607525*^9, 3.8274232119235067`*^9}, 3.8274233238750467`*^9, {
   3.827423970266508*^9, 3.827423998925262*^9}, 3.827426379642764*^9, {
   3.827426605573079*^9, 3.8274266059248323`*^9}, {3.827426670259122*^9, 
   3.8274266756457357`*^9}, {3.827426791863283*^9, 3.8274268694073143`*^9}, 
   3.827426946754819*^9, {3.8274270194853497`*^9, 3.827427022348113*^9}, {
   3.8274270534951153`*^9, 3.8274270568842163`*^9}, 3.827427113217507*^9, 
   3.8274273523615437`*^9, {3.827427734263036*^9, 3.827427736065769*^9}, {
   3.829049415138986*^9, 3.8290494878718987`*^9}, {3.8363289366296597`*^9, 
   3.836329288866592*^9}, {3.8364086901336184`*^9, 3.836408690241274*^9}, {
   3.836408869186132*^9, 3.836409388538095*^9}, {3.836409425027566*^9, 
   3.836409523268422*^9}, {3.836409569913865*^9, 3.836409578709041*^9}, {
   3.836409779073736*^9, 3.8364097812032557`*^9}, {3.836409894544628*^9, 
   3.8364098986537933`*^9}, {3.8364099481662073`*^9, 3.836409960120277*^9}, {
   3.836410026312935*^9, 3.836410041203286*^9}},
 CellLabel->"In[1]:=",ExpressionUUID->"54c7e123-2df9-4985-84a2-ba710761ccef"]
}, Open  ]],

Cell[CellGroupData[{

Cell["Check the constraints:", "Subsubsection",
 CellChangeTimes->{{3.827422138442821*^9, 
  3.8274221402879143`*^9}},ExpressionUUID->"007f87ba-375b-4d63-b7c1-\
4b17b1348797"],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{"u", "[", 
    RowBox[{"x", ",", "y", ",", "0", ",", "gu"}], "]"}], "\[Equal]", 
   RowBox[{"u", "[", 
    RowBox[{"x", ",", "y", ",", "1", ",", "gu"}], "]"}]}], "]"}]], "Input",
 CellChangeTimes->{{3.827424318077579*^9, 3.827424337844503*^9}, {
  3.827426463555253*^9, 3.8274264672917757`*^9}, {3.836329297884487*^9, 
  3.8363293064550734`*^9}, {3.836409406526845*^9, 3.8364094128712473`*^9}},
 CellLabel->"In[19]:=",ExpressionUUID->"63b4e5d3-5fc6-4ba8-80aa-5f96f330cd64"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.829049661836315*^9, 3.829050083458933*^9, 3.836329379094112*^9, 
   3.836329468220337*^9, {3.836408693364077*^9, 3.8364087175426073`*^9}, 
   3.836409413921021*^9, 3.836409453193016*^9, 3.836409978195751*^9, 
   3.836410045548398*^9, 3.836410306989567*^9},
 CellLabel->"Out[19]=",ExpressionUUID->"fe20d64b-e50f-4a87-b160-ddaf5446ec3f"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{
    RowBox[{"u", "[", 
     RowBox[{"x", ",", "0", ",", "z", ",", "gu"}], "]"}], "+", 
    RowBox[{"v", "[", 
     RowBox[{"x", ",", "0", ",", "z", ",", "gu", ",", "gv"}], "]"}]}], 
   "\[Equal]", "5"}], "]"}]], "Input",
 CellChangeTimes->{{3.8274243473507643`*^9, 3.827424353564328*^9}, {
  3.8274244066385393`*^9, 3.8274244069234047`*^9}, {3.827427082966717*^9, 
  3.827427090159446*^9}, {3.836329310389545*^9, 3.8363293194433947`*^9}, {
  3.836409989593174*^9, 3.836410006713183*^9}},
 CellLabel->"In[20]:=",ExpressionUUID->"66378517-0e3e-4419-a32d-546249ada417"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.827424353929631*^9, 3.82742440763645*^9, {3.827426400810358*^9, 
   3.8274264131488953`*^9}, 3.827426688686427*^9, {3.827427078689726*^9, 
   3.827427090934602*^9}, 3.829047812527581*^9, 3.8290496619616747`*^9, 
   3.829050083582625*^9, 3.83632937743342*^9, 3.836329468243326*^9, {
   3.83640869719466*^9, 3.836408722011682*^9}, 3.836410007658695*^9, 
   3.836410045667811*^9, 3.836410307106722*^9},
 CellLabel->"Out[20]=",ExpressionUUID->"24533541-3bb0-4558-9256-aa334d5736f9"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{
    RowBox[{"v", "[", 
     RowBox[{"x", ",", "1", ",", "z", ",", "gu", ",", "gv"}], "]"}], "+", 
    RowBox[{"(", 
     RowBox[{
      RowBox[{"D", "[", 
       RowBox[{
        RowBox[{"v", "[", 
         RowBox[{"x", ",", "y", ",", "z", ",", "gu", ",", "gv"}], "]"}], ",", 
        "y"}], "]"}], "/.", 
      RowBox[{"y", "\[Rule]", "1"}]}], ")"}]}], "\[Equal]", "\[Pi]"}], 
  "]"}]], "Input",
 CellChangeTimes->{{3.827422146228263*^9, 3.827422156181967*^9}, {
   3.827422359723295*^9, 3.8274223598271646`*^9}, {3.827422564839365*^9, 
   3.827422588057219*^9}, {3.827422632822392*^9, 3.8274226381299667`*^9}, 
   3.827423243947611*^9, {3.827423288964285*^9, 3.827423302810828*^9}, {
   3.8274234017691298`*^9, 3.827423401990749*^9}, {3.827423512130094*^9, 
   3.8274235123808413`*^9}, {3.827423561449224*^9, 3.827423564162545*^9}, 
   3.827423794275158*^9, 3.827423845017529*^9, {3.8274240463612843`*^9, 
   3.827424046533657*^9}, {3.827424096697818*^9, 3.827424111720759*^9}, {
   3.827424192421815*^9, 3.8274242263645153`*^9}, {3.827424262526602*^9, 
   3.827424285297472*^9}, {3.827426447052047*^9, 3.827426458509675*^9}, {
   3.827426634145176*^9, 3.827426635319694*^9}, {3.827427095725871*^9, 
   3.827427101282956*^9}, {3.827427623107966*^9, 3.8274276233622313`*^9}, {
   3.836329324365878*^9, 3.836329367006427*^9}, {3.836410057738874*^9, 
   3.83641008784231*^9}},
 CellLabel->"In[21]:=",ExpressionUUID->"05ff9fea-6138-45f1-8e8d-d3d199f22d2f"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{3.8364100888922653`*^9, 3.836410307221538*^9},
 CellLabel->"Out[21]=",ExpressionUUID->"220669f3-7e4a-4b3f-8aa8-13f705a60b2e"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{"Integrate", "[", 
    RowBox[{
     RowBox[{"Integrate", "[", 
      RowBox[{
       RowBox[{"u", "[", 
        RowBox[{"1", ",", "y", ",", "z", ",", "gu"}], "]"}], ",", 
       RowBox[{"{", 
        RowBox[{"y", ",", "0", ",", "1"}], "}"}]}], "]"}], ",", 
     RowBox[{"{", 
      RowBox[{"z", ",", "0", ",", "2"}], "}"}]}], "]"}], "\[Equal]", "1"}], 
  "]"}]], "Input",
 CellChangeTimes->{{3.836329382620902*^9, 3.83632940664517*^9}, {
  3.8364101065477657`*^9, 3.836410135837566*^9}},
 CellLabel->"In[22]:=",ExpressionUUID->"44529084-308c-450d-83b1-accdce4b65d4"],

Cell[BoxData[
 RowBox[{
  RowBox[{
   SubsuperscriptBox["\[Integral]", "0", "2"], 
   RowBox[{
    RowBox[{"(", 
     RowBox[{
      SubsuperscriptBox["\[Integral]", "0", "1"], 
      RowBox[{
       RowBox[{"(", 
        RowBox[{
         FractionBox["1", "2"], "+", 
         RowBox[{
          RowBox[{"(", 
           RowBox[{
            RowBox[{"-", "1"}], "+", "z"}], ")"}], " ", 
          RowBox[{"gu", "[", 
           RowBox[{"1", ",", "y", ",", "0"}], "]"}]}], "-", 
         RowBox[{
          RowBox[{"(", 
           RowBox[{
            RowBox[{"-", "1"}], "+", "z"}], ")"}], " ", 
          RowBox[{"gu", "[", 
           RowBox[{"1", ",", "y", ",", "1"}], "]"}]}], "+", 
         RowBox[{"gu", "[", 
          RowBox[{"1", ",", "y", ",", "z"}], "]"}], "-", 
         RowBox[{
          FractionBox["1", "2"], " ", 
          RowBox[{
           SubsuperscriptBox["\[Integral]", "0", "2"], 
           RowBox[{
            RowBox[{"(", 
             RowBox[{
              SubsuperscriptBox["\[Integral]", "0", "1"], 
              RowBox[{
               RowBox[{"gu", "[", 
                RowBox[{"1", ",", "\[Alpha]", ",", "\[Beta]"}], "]"}], 
               RowBox[{"\[DifferentialD]", "\[Alpha]"}]}]}], ")"}], 
            RowBox[{"\[DifferentialD]", "\[Beta]"}]}]}]}]}], ")"}], 
       RowBox[{"\[DifferentialD]", "y"}]}]}], ")"}], 
    RowBox[{"\[DifferentialD]", "z"}]}]}], "\[Equal]", "1"}]], "Output",
 CellChangeTimes->{
  3.836329407085926*^9, 3.836329468311013*^9, {3.836408697337728*^9, 
   3.836408722240727*^9}, 3.836410045748193*^9, {3.8364101301712017`*^9, 
   3.8364101401664762`*^9}, 3.83641031376871*^9},
 CellLabel->"Out[22]=",ExpressionUUID->"8a02597e-9faf-44c7-ae55-8b779ca15db9"]
}, Open  ]],

Cell["\<\
This final expression can be shown to be true by first showing that the terms \
with (-1+z) go to zero,\
\>", "Text",
 CellChangeTimes->{{3.836410175869987*^9, 
  3.8364102239753447`*^9}},ExpressionUUID->"74fc00f2-04b9-4b45-b9a8-\
4f25939ef459"],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"FullSimplify", "[", 
  RowBox[{"Integrate", "[", 
   RowBox[{
    RowBox[{"Integrate", "[", 
     RowBox[{
      RowBox[{
       RowBox[{"-", "1"}], "+", "z"}], ",", 
      RowBox[{"{", 
       RowBox[{"y", ",", "0", ",", "1"}], "}"}]}], "]"}], ",", 
    RowBox[{"{", 
     RowBox[{"z", ",", "0", ",", "2"}], "}"}]}], "]"}], "]"}]], "Input",
 CellChangeTimes->{{3.836410194737494*^9, 3.836410218788994*^9}},
 CellLabel->"In[23]:=",ExpressionUUID->"3e22d009-9080-4030-b501-c04156309ca2"],

Cell[BoxData["0"], "Output",
 CellChangeTimes->{{3.8364102026753063`*^9, 3.8364102192709227`*^9}, 
   3.836410313908103*^9},
 CellLabel->"Out[23]=",ExpressionUUID->"673665b1-c242-4528-9632-18f8b3bc04c2"]
}, Open  ]],

Cell["and then using", "Text",
 CellChangeTimes->{{3.836410227654086*^9, 
  3.836410247693809*^9}},ExpressionUUID->"6cbb0ac7-39bb-4bba-866b-\
1725800152b2"],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"FullSimplify", "[", 
  RowBox[{"Integrate", "[", 
   RowBox[{
    RowBox[{"Integrate", "[", 
     RowBox[{
      RowBox[{"1", "/", "2"}], ",", 
      RowBox[{"{", 
       RowBox[{"y", ",", "0", ",", "1"}], "}"}]}], "]"}], ",", 
    RowBox[{"{", 
     RowBox[{"z", ",", "0", ",", "2"}], "}"}]}], "]"}], "]"}]], "Input",
 CellChangeTimes->{{3.836410252094626*^9, 3.836410252349606*^9}},
 CellLabel->"In[24]:=",ExpressionUUID->"549febdf-b607-453e-a83d-f2be59e6e69a"],

Cell[BoxData["1"], "Output",
 CellChangeTimes->{3.83641025275526*^9, 3.8364103139448633`*^9},
 CellLabel->"Out[24]=",ExpressionUUID->"7942844a-7c32-469a-9da0-340e38f60ac7"]
}, Open  ]],

Cell["to reduce the expression further and prove it is true. ", "Text",
 CellChangeTimes->{{3.8364102580779467`*^9, 
  3.836410265101658*^9}},ExpressionUUID->"591e8ce3-a562-4ab7-a334-\
d45ab256c816"]
}, Open  ]],

Cell[CellGroupData[{

Cell["Show the constrained expression:", "Subsubsection",
 CellChangeTimes->{{3.836410273182068*^9, 
  3.836410275796474*^9}},ExpressionUUID->"37a25cd0-51e7-49c8-864d-\
c0479c7930a3"],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"u", "[", 
  RowBox[{"x", ",", "y", ",", "z", ",", "gu"}], "]"}]], "Input",
 CellChangeTimes->{{3.8364102830413647`*^9, 3.836410285050886*^9}},
 CellLabel->"In[25]:=",ExpressionUUID->"53a22c90-6504-480c-86f2-e58b7a00b4f8"],

Cell[BoxData[
 RowBox[{
  RowBox[{"gu", "[", 
   RowBox[{"x", ",", "y", ",", "z"}], "]"}], "+", 
  RowBox[{
   RowBox[{"(", 
    RowBox[{"1", "-", "z"}], ")"}], " ", 
   RowBox[{"(", 
    RowBox[{
     RowBox[{"-", 
      RowBox[{"gu", "[", 
       RowBox[{"x", ",", "y", ",", "0"}], "]"}]}], "+", 
     RowBox[{"gu", "[", 
      RowBox[{"x", ",", "y", ",", "1"}], "]"}], "+", 
     RowBox[{
      FractionBox["1", "2"], " ", 
      RowBox[{"(", 
       RowBox[{"1", "-", 
        RowBox[{
         SubsuperscriptBox["\[Integral]", "0", "2"], 
         RowBox[{
          RowBox[{"(", 
           RowBox[{
            SubsuperscriptBox["\[Integral]", "0", "1"], 
            RowBox[{
             RowBox[{"gu", "[", 
              RowBox[{"x", ",", "\[Alpha]", ",", "\[Beta]"}], "]"}], 
             RowBox[{"\[DifferentialD]", "\[Alpha]"}]}]}], ")"}], 
          RowBox[{"\[DifferentialD]", "\[Beta]"}]}]}]}], ")"}]}], "+", 
     RowBox[{
      FractionBox["1", "2"], " ", 
      RowBox[{"(", 
       RowBox[{
        RowBox[{"-", "1"}], "+", 
        RowBox[{
         SubsuperscriptBox["\[Integral]", "0", "2"], 
         RowBox[{
          RowBox[{"(", 
           RowBox[{
            SubsuperscriptBox["\[Integral]", "0", "1"], 
            RowBox[{
             RowBox[{"gu", "[", 
              RowBox[{"x", ",", "\[Alpha]", ",", "\[Beta]"}], "]"}], 
             RowBox[{"\[DifferentialD]", "\[Alpha]"}]}]}], ")"}], 
          RowBox[{"\[DifferentialD]", "\[Beta]"}]}]}]}], ")"}]}]}], ")"}]}], 
  "+", 
  RowBox[{
   FractionBox["1", "2"], " ", 
   RowBox[{"(", 
    RowBox[{"1", "-", 
     RowBox[{
      SubsuperscriptBox["\[Integral]", "0", "2"], 
      RowBox[{
       RowBox[{"(", 
        RowBox[{
         SubsuperscriptBox["\[Integral]", "0", "1"], 
         RowBox[{
          RowBox[{"gu", "[", 
           RowBox[{"x", ",", "\[Alpha]", ",", "\[Beta]"}], "]"}], 
          RowBox[{"\[DifferentialD]", "\[Alpha]"}]}]}], ")"}], 
       RowBox[{"\[DifferentialD]", "\[Beta]"}]}]}]}], ")"}]}]}]], "Output",
 CellChangeTimes->{{3.8364102855563173`*^9, 3.836410314023642*^9}},
 CellLabel->"Out[25]=",ExpressionUUID->"627164a8-d4e1-4d4f-976c-7b9e1d6adf4b"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"v", "[", 
  RowBox[{"x", ",", "y", ",", "z", ",", "gu", ",", "gv"}], "]"}]], "Input",
 CellChangeTimes->{{3.836410287248568*^9, 3.8364102925768423`*^9}},
 CellLabel->"In[26]:=",ExpressionUUID->"b72ce777-4ceb-4aa5-9f63-b70c52b661a0"],

Cell[BoxData[
 RowBox[{
  RowBox[{"gv", "[", 
   RowBox[{"x", ",", "y", ",", "z"}], "]"}], "+", 
  RowBox[{
   RowBox[{"(", 
    RowBox[{"1", "-", 
     FractionBox["y", "2"]}], ")"}], " ", 
   RowBox[{"(", 
    RowBox[{"5", "-", 
     RowBox[{"gu", "[", 
      RowBox[{"x", ",", "0", ",", "z"}], "]"}], "-", 
     RowBox[{"gv", "[", 
      RowBox[{"x", ",", "0", ",", "z"}], "]"}], "-", 
     RowBox[{
      RowBox[{"(", 
       RowBox[{"1", "-", "z"}], ")"}], " ", 
      RowBox[{"(", 
       RowBox[{
        RowBox[{"-", 
         RowBox[{"gu", "[", 
          RowBox[{"x", ",", "0", ",", "0"}], "]"}]}], "+", 
        RowBox[{"gu", "[", 
         RowBox[{"x", ",", "0", ",", "1"}], "]"}], "+", 
        RowBox[{
         FractionBox["1", "2"], " ", 
         RowBox[{"(", 
          RowBox[{"1", "-", 
           RowBox[{
            SubsuperscriptBox["\[Integral]", "0", "2"], 
            RowBox[{
             RowBox[{"(", 
              RowBox[{
               SubsuperscriptBox["\[Integral]", "0", "1"], 
               RowBox[{
                RowBox[{"gu", "[", 
                 RowBox[{"x", ",", "\[Alpha]", ",", "\[Beta]"}], "]"}], 
                RowBox[{"\[DifferentialD]", "\[Alpha]"}]}]}], ")"}], 
             RowBox[{"\[DifferentialD]", "\[Beta]"}]}]}]}], ")"}]}], "+", 
        RowBox[{
         FractionBox["1", "2"], " ", 
         RowBox[{"(", 
          RowBox[{
           RowBox[{"-", "1"}], "+", 
           RowBox[{
            SubsuperscriptBox["\[Integral]", "0", "2"], 
            RowBox[{
             RowBox[{"(", 
              RowBox[{
               SubsuperscriptBox["\[Integral]", "0", "1"], 
               RowBox[{
                RowBox[{"gu", "[", 
                 RowBox[{"x", ",", "\[Alpha]", ",", "\[Beta]"}], "]"}], 
                RowBox[{"\[DifferentialD]", "\[Alpha]"}]}]}], ")"}], 
             RowBox[{"\[DifferentialD]", "\[Beta]"}]}]}]}], ")"}]}]}], 
       ")"}]}], "+", 
     RowBox[{
      FractionBox["1", "2"], " ", 
      RowBox[{"(", 
       RowBox[{
        RowBox[{"-", "1"}], "+", 
        RowBox[{
         SubsuperscriptBox["\[Integral]", "0", "2"], 
         RowBox[{
          RowBox[{"(", 
           RowBox[{
            SubsuperscriptBox["\[Integral]", "0", "1"], 
            RowBox[{
             RowBox[{"gu", "[", 
              RowBox[{"x", ",", "\[Alpha]", ",", "\[Beta]"}], "]"}], 
             RowBox[{"\[DifferentialD]", "\[Alpha]"}]}]}], ")"}], 
          RowBox[{"\[DifferentialD]", "\[Beta]"}]}]}]}], ")"}]}]}], ")"}]}], 
  "+", 
  RowBox[{
   FractionBox["1", "2"], " ", "y", " ", 
   RowBox[{"(", 
    RowBox[{"\[Pi]", "-", 
     RowBox[{"gv", "[", 
      RowBox[{"x", ",", "1", ",", "z"}], "]"}], "-", 
     RowBox[{
      SuperscriptBox["gv", 
       TagBox[
        RowBox[{"(", 
         RowBox[{"0", ",", "1", ",", "0"}], ")"}],
        Derivative],
       MultilineFunction->None], "[", 
      RowBox[{"x", ",", "1", ",", "z"}], "]"}]}], ")"}]}]}]], "Output",
 CellChangeTimes->{{3.836410292851944*^9, 3.836410314098151*^9}},
 CellLabel->"Out[26]=",ExpressionUUID->"1cb6f544-4ed4-4f6f-b4eb-f7481d04c4a8"]
}, Open  ]]
}, Open  ]]
},
WindowSize->{1360, 704},
WindowMargins->{{0, Automatic}, {0, Automatic}},
Magnification:>1.5 Inherited,
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
Cell[1510, 35, 187, 3, 69, "Subsubsection",ExpressionUUID->"c9b673bf-907a-4716-9647-816a981a0b8d"],
Cell[1700, 40, 7332, 201, 728, "Input",ExpressionUUID->"54c7e123-2df9-4985-84a2-ba710761ccef"]
}, Open  ]],
Cell[CellGroupData[{
Cell[9069, 246, 175, 3, 69, "Subsubsection",ExpressionUUID->"007f87ba-375b-4d63-b7c1-4b17b1348797"],
Cell[CellGroupData[{
Cell[9269, 253, 540, 10, 47, "Input",ExpressionUUID->"63b4e5d3-5fc6-4ba8-80aa-5f96f330cd64"],
Cell[9812, 265, 391, 6, 53, "Output",ExpressionUUID->"fe20d64b-e50f-4a87-b160-ddaf5446ec3f"]
}, Open  ]],
Cell[CellGroupData[{
Cell[10240, 276, 634, 13, 47, "Input",ExpressionUUID->"66378517-0e3e-4419-a32d-546249ada417"],
Cell[10877, 291, 534, 8, 53, "Output",ExpressionUUID->"24533541-3bb0-4558-9256-aa334d5736f9"]
}, Open  ]],
Cell[CellGroupData[{
Cell[11448, 304, 1522, 29, 47, "Input",ExpressionUUID->"05ff9fea-6138-45f1-8e8d-d3d199f22d2f"],
Cell[12973, 335, 176, 2, 53, "Output",ExpressionUUID->"220669f3-7e4a-4b3f-8aa8-13f705a60b2e"]
}, Open  ]],
Cell[CellGroupData[{
Cell[13186, 342, 634, 16, 47, "Input",ExpressionUUID->"44529084-308c-450d-83b1-accdce4b65d4"],
Cell[13823, 360, 1722, 45, 115, "Output",ExpressionUUID->"8a02597e-9faf-44c7-ae55-8b779ca15db9"]
}, Open  ]],
Cell[15560, 408, 255, 6, 54, "Text",ExpressionUUID->"74fc00f2-04b9-4b45-b9a8-4f25939ef459"],
Cell[CellGroupData[{
Cell[15840, 418, 510, 13, 47, "Input",ExpressionUUID->"3e22d009-9080-4030-b501-c04156309ca2"],
Cell[16353, 433, 203, 3, 53, "Output",ExpressionUUID->"673665b1-c242-4528-9632-18f8b3bc04c2"]
}, Open  ]],
Cell[16571, 439, 156, 3, 54, "Text",ExpressionUUID->"6cbb0ac7-39bb-4bba-866b-1725800152b2"],
Cell[CellGroupData[{
Cell[16752, 446, 487, 12, 47, "Input",ExpressionUUID->"549febdf-b607-453e-a83d-f2be59e6e69a"],
Cell[17242, 460, 172, 2, 53, "Output",ExpressionUUID->"7942844a-7c32-469a-9da0-340e38f60ac7"]
}, Open  ]],
Cell[17429, 465, 199, 3, 54, "Text",ExpressionUUID->"591e8ce3-a562-4ab7-a334-d45ab256c816"]
}, Open  ]],
Cell[CellGroupData[{
Cell[17665, 473, 183, 3, 69, "Subsubsection",ExpressionUUID->"37a25cd0-51e7-49c8-864d-c0479c7930a3"],
Cell[CellGroupData[{
Cell[17873, 480, 245, 4, 47, "Input",ExpressionUUID->"53a22c90-6504-480c-86f2-e58b7a00b4f8"],
Cell[18121, 486, 2168, 62, 177, "Output",ExpressionUUID->"627164a8-d4e1-4d4f-976c-7b9e1d6adf4b"]
}, Open  ]],
Cell[CellGroupData[{
Cell[20326, 553, 256, 4, 47, "Input",ExpressionUUID->"b72ce777-4ceb-4aa5-9f63-b70c52b661a0"],
Cell[20585, 559, 3100, 88, 292, "Output",ExpressionUUID->"1cb6f544-4ed4-4f6f-b4eb-f7481d04c4a8"]
}, Open  ]]
}, Open  ]]
}
]
*)

(* End of internal cache information *)

(* NotebookSignature CxTmOyFikEf0kAwLBLrkM6tM *)
