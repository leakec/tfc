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
NotebookDataLength[     23301,        601]
NotebookOptionsPosition[     21763,        566]
NotebookOutlinePosition[     22126,        582]
CellTagsIndexPosition[     22083,        579]
WindowFrame->Normal*)

(* Beginning of Notebook Content *)
Notebook[{
Cell["Setup finite fields", "Text",
 CellChangeTimes->{{3.825587407759284*^9, 
  3.8255874117270403`*^9}},ExpressionUUID->"245417f8-1d0a-4ab6-b122-\
c8a14381fe5a"],

Cell[BoxData[{
 RowBox[{
  RowBox[{"<<", "FiniteFields`"}], " ", 
  RowBox[{"(*", " ", 
   RowBox[{"Load", " ", "finite", " ", "field", " ", "package"}], " ", 
   "*)"}]}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"SetFieldFormat", "[", 
   RowBox[{
    RowBox[{"GF", "[", 
     RowBox[{"2", ",", "2"}], "]"}], ",", 
    RowBox[{"FormatType", "\[Rule]", 
     RowBox[{"FunctionOfCode", "[", "F", "]"}]}]}], "]"}], 
  RowBox[{"(*", " ", 
   RowBox[{
   "Set", " ", "F", " ", "as", " ", "field", " ", "with", " ", "4", " ", 
    "elements"}], " ", "*)"}]}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"Output", "[", "x_", "]"}], ":=", 
    RowBox[{"x", "/.", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{
        RowBox[{"F", "[", "1", "]"}], "\[Rule]", "1"}], ",", 
       RowBox[{
        RowBox[{"F", "[", "2", "]"}], "\[Rule]", "A"}], ",", 
       RowBox[{
        RowBox[{"F", "[", "3", "]"}], "\[Rule]", "B"}]}], "}"}]}]}], ";"}], 
  " ", 
  RowBox[{"(*", " ", 
   RowBox[{
   "Function", " ", "to", " ", "make", " ", "field", " ", "elements", " ", 
    RowBox[{"{", 
     RowBox[{"0", ",", "1", ",", "A", ",", "B"}], "}"}]}], " ", 
   "*)"}]}]}], "Input",
 CellChangeTimes->{{3.8255856081593513`*^9, 3.825585679526134*^9}, 
   3.825585774524634*^9, {3.825585870573354*^9, 3.825585873485368*^9}, {
   3.8255859684675198`*^9, 3.825585969089682*^9}, {3.825586082488052*^9, 
   3.825586117683979*^9}, {3.825586152970738*^9, 3.825586155879689*^9}, {
   3.8255861990693207`*^9, 3.8255862398440437`*^9}, {3.825586281492217*^9, 
   3.82558629621598*^9}, {3.825586334709455*^9, 3.825586341991748*^9}, {
   3.825586524740995*^9, 3.825586538079629*^9}, {3.8255865860249653`*^9, 
   3.825586643663273*^9}, {3.825586723865533*^9, 3.825586742611919*^9}, {
   3.825586871384522*^9, 3.82558689934515*^9}, {3.82558695530272*^9, 
   3.825586963110447*^9}, {3.825587254025794*^9, 3.825587263616386*^9}, {
   3.825587321566843*^9, 3.825587404063485*^9}, {3.8255877004733133`*^9, 
   3.825587712639761*^9}},
 CellLabel->"In[1]:=",ExpressionUUID->"26533c4d-02e3-4829-b4cf-f42315daa595"],

Cell[CellGroupData[{

Cell["Addition and multiplication tables:", "Subsubsection",
 CellChangeTimes->{{3.825587421319215*^9, 3.8255874237103024`*^9}, {
  3.826134428115203*^9, 
  3.826134433963874*^9}},ExpressionUUID->"e4cd5fa0-3ab7-4e21-a2ea-\
7872c52312c5"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{"elements", " ", "=", " ", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"F", "[", "0", "]"}], ",", 
     RowBox[{"F", "[", "1", "]"}], ",", 
     RowBox[{"F", "[", "2", "]"}], ",", 
     RowBox[{"F", "[", "3", "]"}]}], "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"fullElements", " ", "=", " ", 
   RowBox[{"{", 
    RowBox[{"elements", ",", "elements", ",", "elements", ",", "elements"}], 
    "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"plus", " ", "=", " ", 
   RowBox[{"fullElements", "+", 
    RowBox[{"Transpose", "[", "fullElements", "]"}]}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"multiply", " ", "=", " ", 
   RowBox[{"fullElements", "*", 
    RowBox[{"Transpose", "[", "fullElements", "]"}]}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"plusTab", " ", "=", " ", 
   RowBox[{
    RowBox[{"ArrayFlatten", "[", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{"{", 
        RowBox[{"\"\<+\>\"", ",", 
         RowBox[{"{", "elements", "}"}]}], "}"}], ",", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"Transpose", "[", 
          RowBox[{"{", "elements", "}"}], "]"}], ",", "plus"}], "}"}]}], 
      "}"}], "]"}], "//", "Output"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"multiplyTab", " ", "=", " ", 
   RowBox[{
    RowBox[{"ArrayFlatten", "[", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{"{", 
        RowBox[{"\"\<*\>\"", ",", 
         RowBox[{"{", "elements", "}"}]}], "}"}], ",", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"Transpose", "[", 
          RowBox[{"{", "elements", "}"}], "]"}], ",", "multiply"}], "}"}]}], 
      "}"}], "]"}], "//", "Output"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{"Grid", "[", 
  RowBox[{"plusTab", ",", 
   RowBox[{"Alignment", "\[Rule]", "Left"}], ",", 
   RowBox[{"Spacings", "\[Rule]", 
    RowBox[{"{", 
     RowBox[{"2", ",", "1"}], "}"}]}], ",", 
   RowBox[{"Frame", "\[Rule]", "All"}], ",", 
   RowBox[{"ItemStyle", "\[Rule]", "\"\<Text\>\""}], ",", 
   RowBox[{"Background", "\[Rule]", 
    RowBox[{"{", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{"Gray", ",", "None"}], "}"}], ",", 
      RowBox[{"{", 
       RowBox[{"LightGray", ",", "None"}], "}"}]}], "}"}]}]}], 
  "]"}], "\[IndentingNewLine]", 
 RowBox[{"Grid", "[", 
  RowBox[{"multiplyTab", ",", 
   RowBox[{"Alignment", "\[Rule]", "Left"}], ",", 
   RowBox[{"Spacings", "\[Rule]", 
    RowBox[{"{", 
     RowBox[{"2", ",", "1"}], "}"}]}], ",", 
   RowBox[{"Frame", "\[Rule]", "All"}], ",", 
   RowBox[{"ItemStyle", "\[Rule]", "\"\<Text\>\""}], ",", 
   RowBox[{"Background", "\[Rule]", 
    RowBox[{"{", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{"Gray", ",", "None"}], "}"}], ",", 
      RowBox[{"{", 
       RowBox[{"LightGray", ",", "None"}], "}"}]}], "}"}]}]}], 
  "]"}]}], "Input",
 CellChangeTimes->{{3.8255878001877823`*^9, 3.825587828189024*^9}, {
   3.825587980819427*^9, 3.825587986701118*^9}, 3.825588067319592*^9, {
   3.825588116299156*^9, 3.825588392310314*^9}, {3.8255884780730267`*^9, 
   3.825588479284746*^9}, {3.825588649911887*^9, 3.825588655486874*^9}, {
   3.825588696501259*^9, 3.82558869962394*^9}, {3.825589066067792*^9, 
   3.82558908341003*^9}, 3.825589120085288*^9, {3.825589226444214*^9, 
   3.825589255499447*^9}, {3.825589306867848*^9, 3.825589342490386*^9}, {
   3.825589387585856*^9, 3.8255894446028*^9}, 3.825590862991172*^9},
 CellLabel->"In[4]:=",ExpressionUUID->"d8902316-fa07-475a-9cf5-4262eb4f21aa"],

Cell[BoxData[
 TagBox[GridBox[{
    {"\<\"+\"\>", "0", "1", "A", "B"},
    {"0", "0", "1", "A", "B"},
    {"1", "1", "0", "B", "A"},
    {"A", "A", "B", "0", "1"},
    {"B", "B", "A", "1", "0"}
   },
   AutoDelete->False,
   GridBoxAlignment->{"Columns" -> {{Left}}},
   GridBoxBackground->{"Columns" -> {
       GrayLevel[0.5], None}, "Rows" -> {
       GrayLevel[0.85], None}},
   GridBoxFrame->{"Columns" -> {{True}}, "Rows" -> {{True}}},
   GridBoxItemSize->{"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}},
   GridBoxItemStyle->{"Columns" -> {{"Text"}}, "Rows" -> {{"Text"}}},
   GridBoxSpacings->{"Columns" -> {{2}}, "Rows" -> {{1}}}],
  "Grid"]], "Output",
 CellChangeTimes->{{3.8255882322924423`*^9, 3.825588246040955*^9}, 
   3.8255882824112597`*^9, 3.825588369967259*^9, 3.825588479944271*^9, 
   3.8255886567291317`*^9, 3.825588700341145*^9, 3.825589084667316*^9, 
   3.825589120422714*^9, {3.825589234816945*^9, 3.82558925659966*^9}, 
   3.8255893180630007`*^9, {3.825589422197546*^9, 3.825589445546691*^9}, 
   3.825589580523068*^9, 3.8256096020208178`*^9, 3.825611430535247*^9, 
   3.8256125326258173`*^9, 3.8256127563845*^9, 3.8261107377202272`*^9, 
   3.826134391619185*^9, 3.8261344363875303`*^9, 3.829830597078141*^9},
 CellLabel->"Out[10]=",ExpressionUUID->"c2972005-4976-4eb0-8ef4-348c0d1ae24c"],

Cell[BoxData[
 TagBox[GridBox[{
    {"\<\"*\"\>", "0", "1", "A", "B"},
    {"0", "0", "0", "0", "0"},
    {"1", "0", "1", "A", "B"},
    {"A", "0", "A", "B", "1"},
    {"B", "0", "B", "1", "A"}
   },
   AutoDelete->False,
   GridBoxAlignment->{"Columns" -> {{Left}}},
   GridBoxBackground->{"Columns" -> {
       GrayLevel[0.5], None}, "Rows" -> {
       GrayLevel[0.85], None}},
   GridBoxFrame->{"Columns" -> {{True}}, "Rows" -> {{True}}},
   GridBoxItemSize->{"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}},
   GridBoxItemStyle->{"Columns" -> {{"Text"}}, "Rows" -> {{"Text"}}},
   GridBoxSpacings->{"Columns" -> {{2}}, "Rows" -> {{1}}}],
  "Grid"]], "Output",
 CellChangeTimes->{{3.8255882322924423`*^9, 3.825588246040955*^9}, 
   3.8255882824112597`*^9, 3.825588369967259*^9, 3.825588479944271*^9, 
   3.8255886567291317`*^9, 3.825588700341145*^9, 3.825589084667316*^9, 
   3.825589120422714*^9, {3.825589234816945*^9, 3.82558925659966*^9}, 
   3.8255893180630007`*^9, {3.825589422197546*^9, 3.825589445546691*^9}, 
   3.825589580523068*^9, 3.8256096020208178`*^9, 3.825611430535247*^9, 
   3.8256125326258173`*^9, 3.8256127563845*^9, 3.8261107377202272`*^9, 
   3.826134391619185*^9, 3.8261344363875303`*^9, 3.829830597093596*^9},
 CellLabel->"Out[11]=",ExpressionUUID->"b07b0000-5c35-44bf-9be6-2a089f8bc024"]
}, Open  ]]
}, Open  ]],

Cell[CellGroupData[{

Cell["Multivariate constrained expression:", "Subsubsection",
 CellChangeTimes->{{3.825609610376367*^9, 3.825609618601001*^9}, 
   3.826134442612084*^9},ExpressionUUID->"25c94a0d-2b89-4d55-9a52-\
045f39d025d9"],

Cell["Constraints : u(0,y) = A,  u(B,y) = 1,  u(x,0) = u(x,B)", "Text",
 CellChangeTimes->{{3.825609625083358*^9, 
  3.8256096603761683`*^9}},ExpressionUUID->"b60f4d6d-a888-4c9f-8240-\
7b02eb7fba73"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"s1", "[", "x_", "]"}], ":=", 
   RowBox[{"{", 
    RowBox[{"1", ",", "x"}], "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"Sij", " ", "=", " ", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"s1", "[", "0", "]"}], ",", 
     RowBox[{"s1", "[", 
      RowBox[{"F", "[", "3", "]"}], "]"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"\[Alpha]1", "=", 
   RowBox[{"Inverse", "[", "Sij", "]"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"\[Phi]1", "[", "x_", "]"}], ":=", 
   RowBox[{
    RowBox[{"s1", "[", "x", "]"}], ".", "\[Alpha]1"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"\[Rho]1", "[", 
    RowBox[{"x_", ",", "y_", ",", "g_Symbol"}], "]"}], ":=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{
      RowBox[{"F", "[", "2", "]"}], "-", 
      RowBox[{"g", "[", 
       RowBox[{"0", ",", "y"}], "]"}]}], ",", 
     RowBox[{"1", "-", 
      RowBox[{"g", "[", 
       RowBox[{
        RowBox[{"F", "[", "3", "]"}], ",", "y"}], "]"}]}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"u1", "[", 
    RowBox[{"x_", ",", "y_", ",", "g_Symbol"}], "]"}], ":=", 
   RowBox[{
    RowBox[{"g", "[", 
     RowBox[{"x", ",", "y"}], "]"}], "+", 
    RowBox[{
     RowBox[{"\[Phi]1", "[", "x", "]"}], ".", 
     RowBox[{"\[Rho]1", "[", 
      RowBox[{"x", ",", "y", ",", "g"}], "]"}]}]}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"\[Rho]1", "[", 
   RowBox[{"x", ",", "y", ",", "h"}], "]"}], "//", "Output"}]}], "Input",
 CellChangeTimes->{{3.825609766312248*^9, 3.825609811224059*^9}, {
  3.825611068999942*^9, 3.8256110934432983`*^9}, {3.825611148323122*^9, 
  3.825611149363*^9}, {3.825611440566475*^9, 3.825611441594015*^9}, {
  3.825612302073225*^9, 3.825612321761972*^9}, {3.825612353823842*^9, 
  3.82561235655337*^9}, {3.826112801627619*^9, 3.826112810185979*^9}, {
  3.8261128506522217`*^9, 3.826112852203871*^9}, {3.82611619204287*^9, 
  3.826116221843663*^9}},
 CellLabel->"In[12]:=",ExpressionUUID->"f12a1509-0809-46f8-afe3-125a643793e2"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{
   RowBox[{"A", "-", 
    RowBox[{"h", "[", 
     RowBox[{"0", ",", "y"}], "]"}]}], ",", 
   RowBox[{"1", "-", 
    RowBox[{"h", "[", 
     RowBox[{"B", ",", "y"}], "]"}]}]}], "}"}]], "Output",
 CellChangeTimes->{{3.826112803113964*^9, 3.826112810597148*^9}, 
   3.826112852783039*^9, {3.8261161975784082`*^9, 3.826116222069909*^9}, 
   3.826134391657612*^9, 3.826134436418281*^9, 3.829830597121811*^9},
 CellLabel->"Out[18]=",ExpressionUUID->"24cd6eea-403e-4dee-b958-7d78e42fe28d"]
}, Open  ]],

Cell[BoxData[{
 RowBox[{
  RowBox[{
   RowBox[{"s2", "[", "y_", "]"}], ":=", 
   RowBox[{"{", "y", "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"Sij", " ", "=", " ", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"s2", "[", "0", "]"}], "-", 
     RowBox[{"s2", "[", 
      RowBox[{"F", "[", "3", "]"}], "]"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{"\[Alpha]2", "=", 
   RowBox[{"Inverse", "[", "Sij", "]"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"\[Phi]2", "[", "y_", "]"}], ":=", 
   RowBox[{
    RowBox[{"s2", "[", "y", "]"}], ".", "\[Alpha]2"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"\[Rho]2", "[", 
    RowBox[{"x_", ",", "y_", ",", "g_Symbol"}], "]"}], ":=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"u1", "[", 
      RowBox[{"x", ",", 
       RowBox[{"F", "[", "3", "]"}], ",", "g"}], "]"}], "-", 
     RowBox[{"u1", "[", 
      RowBox[{"x", ",", "0", ",", "g"}], "]"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"u", "[", 
    RowBox[{"x_", ",", "y_", ",", "g_Symbol"}], "]"}], ":=", 
   RowBox[{
    RowBox[{"u1", "[", 
     RowBox[{"x", ",", "y", ",", "g"}], "]"}], "+", 
    RowBox[{
     RowBox[{"\[Phi]2", "[", "y", "]"}], ".", 
     RowBox[{"\[Rho]2", "[", 
      RowBox[{"x", ",", "y", ",", "g"}], "]"}]}]}]}], ";"}]}], "Input",
 CellChangeTimes->{{3.825609805498888*^9, 3.825609806484809*^9}, {
  3.825610096382738*^9, 3.825610145614601*^9}, {3.825611011247651*^9, 
  3.825611063899987*^9}, {3.825611099428905*^9, 3.825611146427251*^9}, {
  3.825611179909234*^9, 3.8256111808668613`*^9}, {3.8256112219695997`*^9, 
  3.825611234284955*^9}, {3.825611513770323*^9, 3.825611537785368*^9}, {
  3.825611673603012*^9, 3.825611673697012*^9}},
 CellLabel->"In[19]:=",ExpressionUUID->"39b11562-4d89-404e-91ba-52e02abded9b"],

Cell["Check constraints for any g (x):", "Text",
 CellChangeTimes->{{3.82561130269214*^9, 
  3.8256113184519987`*^9}},ExpressionUUID->"0e58b5f2-8c8b-4f24-831f-\
155271426282"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{"u", "[", 
    RowBox[{"0", ",", "y", ",", "g"}], "]"}], "\[Equal]", 
   RowBox[{"F", "[", "2", "]"}]}], "]"}], "\[IndentingNewLine]", 
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{"u", "[", 
    RowBox[{
     RowBox[{"F", "[", "3", "]"}], ",", "y", ",", "g"}], "]"}], "\[Equal]", 
   "1"}], "]"}], "\[IndentingNewLine]", 
 RowBox[{"FullSimplify", "[", 
  RowBox[{
   RowBox[{"u", "[", 
    RowBox[{"x", ",", "0", ",", "g"}], "]"}], "\[Equal]", 
   RowBox[{"u", "[", 
    RowBox[{"x", ",", 
     RowBox[{"F", "[", "3", "]"}], ",", "g"}], "]"}]}], "]"}]}], "Input",
 CellChangeTimes->{{3.82561131981308*^9, 3.825611329500018*^9}, {
  3.825611552603755*^9, 3.8256115888205643`*^9}, {3.825612748224818*^9, 
  3.825612753522286*^9}},
 CellLabel->"In[25]:=",ExpressionUUID->"91a5585d-6dbe-4d22-ad18-db216f0b8dd7"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.8256113300157433`*^9, {3.825611430760357*^9, 3.825611449330916*^9}, {
   3.8256115484827337`*^9, 3.8256115561296864`*^9}, 3.825611589422255*^9, 
   3.825611677675292*^9, 3.825612532679274*^9, 3.825612756443781*^9, 
   3.826110737781745*^9, 3.8261343917132673`*^9, 3.8261344364649982`*^9, 
   3.8298305971631804`*^9},
 CellLabel->"Out[25]=",ExpressionUUID->"bdf411e0-2a00-4fb0-9aba-c7bd527e8752"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.8256113300157433`*^9, {3.825611430760357*^9, 3.825611449330916*^9}, {
   3.8256115484827337`*^9, 3.8256115561296864`*^9}, 3.825611589422255*^9, 
   3.825611677675292*^9, 3.825612532679274*^9, 3.825612756443781*^9, 
   3.826110737781745*^9, 3.8261343917132673`*^9, 3.8261344364649982`*^9, 
   3.8298305971706*^9},
 CellLabel->"Out[26]=",ExpressionUUID->"aba4df62-7bfe-418e-b8d4-11e8cf72d485"],

Cell[BoxData["True"], "Output",
 CellChangeTimes->{
  3.8256113300157433`*^9, {3.825611430760357*^9, 3.825611449330916*^9}, {
   3.8256115484827337`*^9, 3.8256115561296864`*^9}, 3.825611589422255*^9, 
   3.825611677675292*^9, 3.825612532679274*^9, 3.825612756443781*^9, 
   3.826110737781745*^9, 3.8261343917132673`*^9, 3.8261344364649982`*^9, 
   3.829830597288686*^9},
 CellLabel->"Out[27]=",ExpressionUUID->"7effd9e9-38b2-48da-9902-9d87ff6fbb00"]
}, Open  ]],

Cell["Check constraints for a specific g(x):", "Text",
 CellChangeTimes->{{3.8256116939605303`*^9, 
  3.825611698311619*^9}},ExpressionUUID->"3088c7b8-d4dc-40d5-bc54-\
abb9bf5e0c45"],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", 
   RowBox[{
    RowBox[{
     RowBox[{"g", "[", 
      RowBox[{"x_", ",", "y_"}], "]"}], ":=", 
     RowBox[{"x", "*", "x"}]}], ";"}], "*)"}], "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{
    RowBox[{
     RowBox[{"g", "[", 
      RowBox[{"x_", ",", "y_"}], "]"}], ":=", 
     RowBox[{
      RowBox[{
       RowBox[{"F", "[", "2", "]"}], "*", "x"}], "+", 
      RowBox[{"x", "*", "y"}], "+", "y"}]}], ";"}], "\[IndentingNewLine]", 
   RowBox[{
    RowBox[{"uVals", " ", "=", " ", 
     RowBox[{"Table", "[", 
      RowBox[{
       RowBox[{"u", "[", 
        RowBox[{
         RowBox[{"F", "[", "x", "]"}], ",", 
         RowBox[{"F", "[", "y", "]"}], ",", "g"}], "]"}], ",", 
       RowBox[{"{", 
        RowBox[{"y", ",", "3", ",", "0", ",", 
         RowBox[{"-", "1"}]}], "}"}], ",", 
       RowBox[{"{", 
        RowBox[{"x", ",", "0", ",", "3"}], "}"}]}], "]"}]}], ";"}], 
   "\[IndentingNewLine]", 
   RowBox[{
    RowBox[{"uTab", " ", "=", " ", 
     RowBox[{
      RowBox[{"ArrayFlatten", "[", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"{", 
          RowBox[{
           RowBox[{"Transpose", "[", 
            RowBox[{"{", 
             RowBox[{"Reverse", "[", "elements", "]"}], "}"}], "]"}], ",", 
           "uVals"}], "}"}], ",", 
         RowBox[{"{", 
          RowBox[{"\"\<y/x\>\"", ",", 
           RowBox[{"{", "elements", "}"}]}], "}"}]}], "}"}], "]"}], "//", 
      "Output"}]}], ";"}], "\[IndentingNewLine]", 
   RowBox[{"Grid", "[", 
    RowBox[{"uTab", ",", 
     RowBox[{"Alignment", "\[Rule]", "Left"}], ",", 
     RowBox[{"Spacings", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{"2", ",", "1"}], "}"}]}], ",", 
     RowBox[{"Frame", "\[Rule]", "All"}], ",", 
     RowBox[{"ItemStyle", "\[Rule]", "\"\<Text\>\""}], ",", 
     RowBox[{"Background", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{
        RowBox[{"{", 
         RowBox[{"Gray", ",", "None"}], "}"}], ",", 
        RowBox[{"{", 
         RowBox[{"None", ",", "None", ",", "None", ",", "None", ",", "Gray"}],
          "}"}]}], "}"}]}]}], "]"}]}]}]], "Input",
 CellChangeTimes->{{3.825611907816844*^9, 3.82561195217492*^9}, {
  3.825611991196075*^9, 3.825611995458987*^9}, {3.825612031026606*^9, 
  3.8256120716246967`*^9}, {3.82561210404151*^9, 3.8256121434521523`*^9}, {
  3.825612261075635*^9, 3.825612261194055*^9}, {3.8256123338259974`*^9, 
  3.8256123956884604`*^9}, {3.825612525608914*^9, 3.825612530521449*^9}},
 CellLabel->"In[28]:=",ExpressionUUID->"fdc9ad96-29ca-4344-bd05-e91d2ea49258"],

Cell[BoxData[
 TagBox[GridBox[{
    {"B", "A", "B", "0", "1"},
    {"A", "A", "B", "0", "1"},
    {"1", "A", "B", "0", "1"},
    {"0", "A", "B", "0", "1"},
    {"\<\"y/x\"\>", "0", "1", "A", "B"}
   },
   AutoDelete->False,
   GridBoxAlignment->{"Columns" -> {{Left}}},
   GridBoxBackground->{"Columns" -> {
       GrayLevel[0.5], None}, "Rows" -> {None, None, None, None, 
       GrayLevel[0.5]}},
   GridBoxFrame->{"Columns" -> {{True}}, "Rows" -> {{True}}},
   GridBoxItemSize->{"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}},
   GridBoxItemStyle->{"Columns" -> {{"Text"}}, "Rows" -> {{"Text"}}},
   GridBoxSpacings->{"Columns" -> {{2}}, "Rows" -> {{1}}}],
  "Grid"]], "Output",
 CellChangeTimes->{
  3.825611953701437*^9, {3.825612007440782*^9, 3.825612072011273*^9}, {
   3.825612104525384*^9, 3.825612143708905*^9}, 3.8256122625719852`*^9, {
   3.825612329335841*^9, 3.825612396302162*^9}, 3.8256125328574743`*^9, 
   3.8256127566095552`*^9, 3.82611073799785*^9, 3.82613439187635*^9, 
   3.826134436645617*^9, 3.829830597850877*^9},
 CellLabel->"Out[31]=",ExpressionUUID->"06f32a2e-50a7-4a63-8261-d914345c9cd7"]
}, Open  ]]
}, Open  ]],

Cell[CellGroupData[{

Cell["", "Subsubsection",ExpressionUUID->"48d3cb70-872c-499d-bbc6-805d2e727ca0"],

Cell["", "Text",ExpressionUUID->"d8db177d-48e0-41f6-9b2a-5a2a622b2c07"]
}, Open  ]]
},
WindowSize->{1360, 704},
WindowMargins->{{0, Automatic}, {0, Automatic}},
Magnification:>1.7 Inherited,
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
Cell[1488, 33, 163, 3, 61, "Text",ExpressionUUID->"245417f8-1d0a-4ab6-b122-c8a14381fe5a"],
Cell[1654, 38, 2103, 49, 206, "Input",ExpressionUUID->"26533c4d-02e3-4829-b4cf-f42315daa595"],
Cell[CellGroupData[{
Cell[3782, 91, 237, 4, 78, "Subsubsection",ExpressionUUID->"e4cd5fa0-3ab7-4e21-a2ea-7872c52312c5"],
Cell[CellGroupData[{
Cell[4044, 99, 3512, 92, 472, "Input",ExpressionUUID->"d8902316-fa07-475a-9cf5-4262eb4f21aa"],
Cell[7559, 193, 1320, 26, 284, "Output",ExpressionUUID->"c2972005-4976-4eb0-8ef4-348c0d1ae24c"],
Cell[8882, 221, 1320, 26, 284, "Output",ExpressionUUID->"b07b0000-5c35-44bf-9be6-2a089f8bc024"]
}, Open  ]]
}, Open  ]],
Cell[CellGroupData[{
Cell[10251, 253, 210, 3, 78, "Subsubsection",ExpressionUUID->"25c94a0d-2b89-4d55-9a52-045f39d025d9"],
Cell[10464, 258, 199, 3, 61, "Text",ExpressionUUID->"b60f4d6d-a888-4c9f-8240-7b02eb7fba73"],
Cell[CellGroupData[{
Cell[10688, 265, 2116, 60, 282, "Input",ExpressionUUID->"f12a1509-0809-46f8-afe3-125a643793e2"],
Cell[12807, 327, 522, 12, 85, "Output",ExpressionUUID->"24cd6eea-403e-4dee-b958-7d78e42fe28d"]
}, Open  ]],
Cell[13344, 342, 1863, 51, 244, "Input",ExpressionUUID->"39b11562-4d89-404e-91ba-52e02abded9b"],
Cell[15210, 395, 175, 3, 61, "Text",ExpressionUUID->"0e58b5f2-8c8b-4f24-831f-155271426282"],
Cell[CellGroupData[{
Cell[15410, 402, 884, 22, 130, "Input",ExpressionUUID->"91a5585d-6dbe-4d22-ad18-db216f0b8dd7"],
Cell[16297, 426, 451, 7, 85, "Output",ExpressionUUID->"bdf411e0-2a00-4fb0-9aba-c7bd527e8752"],
Cell[16751, 435, 447, 7, 85, "Output",ExpressionUUID->"aba4df62-7bfe-418e-b8d4-11e8cf72d485"],
Cell[17201, 444, 449, 7, 85, "Output",ExpressionUUID->"7effd9e9-38b2-48da-9902-9d87ff6fbb00"]
}, Open  ]],
Cell[17665, 454, 182, 3, 61, "Text",ExpressionUUID->"3088c7b8-d4dc-40d5-bc54-abb9bf5e0c45"],
Cell[CellGroupData[{
Cell[17872, 461, 2546, 68, 320, "Input",ExpressionUUID->"fdc9ad96-29ca-4344-bd05-e91d2ea49258"],
Cell[20421, 531, 1123, 24, 286, "Output",ExpressionUUID->"06f32a2e-50a7-4a63-8261-d914345c9cd7"]
}, Open  ]]
}, Open  ]],
Cell[CellGroupData[{
Cell[21593, 561, 80, 0, 78, "Subsubsection",ExpressionUUID->"48d3cb70-872c-499d-bbc6-805d2e727ca0"],
Cell[21676, 563, 71, 0, 61, "Text",ExpressionUUID->"d8db177d-48e0-41f6-9b2a-5a2a622b2c07"]
}, Open  ]]
}
]
*)

(* End of internal cache information *)

(* NotebookSignature iw0eWdOLYXL4sBgScvvbqJFN *)
