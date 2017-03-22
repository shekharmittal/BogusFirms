cd "E:\data"
use "annexure_2A2B_quarterly_2014.dta", clear
keep if SaleOrPurchase=="AE"
keep if SalePurchaseType=="CG"|SalePurchaseType=="OT"

replace TransactionType="GD" if TransactionType=="GD "
replace TransactionType="GD" if TransactionType=="GD  "
replace TransactionType="GD" if TransactionType=="GD    "
replace TransactionType="GD" if TransactionType=="GD     "
replace TransactionType="GD" if TransactionType=="GD             "
replace TransactionType="GD" if TransactionType=="Gd"
replace TransactionType="GD" if TransactionType=="gd"
replace TransactionType="GD" if TransactionType=="gD"
replace TransactionType="GD" if TransactionType=="gd "
replace TransactionType="WC" if TransactionType=="wc"
replace TransactionType="WC" if TransactionType=="Wc"
replace TransactionType=trim(TransactionType)


drop T985DF5 T985DF6 T985DF7 T985DF8 FormsStatus T985DF4

replace Rate=trim(Rate)
replace Rate="0" if Rate=="0.0"
replace Rate="0" if Rate=="0.00"
replace Rate="0" if Rate=="00.00"
replace Rate="1" if Rate=="1.00"
replace Rate="1" if Rate=="1.000"
replace Rate="2" if Rate=="2.00"
replace Rate="12.5" if Rate=="12.50"
replace Rate="12.5" if Rate=="12.500"
replace Rate="12.5" if Rate=="12.5000"
replace Rate="20" if Rate=="20.00"
replace Rate="20" if Rate=="20.0"
replace Rate="5" if Rate=="5.0"
replace Rate="5" if Rate=="5.00"
replace Rate="5" if Rate=="5.000"
replace Rate="5" if Rate=="05.00"
replace Rate="4" if Rate=="4.00"
replace Rate="12.5" if Rate=="12.00"
replace Rate="12.5" if Rate=="0.12"
replace Rate="1" if Rate=="0.01"
replace Rate="5" if Rate=="0.05"
replace Rate="20" if Rate=="0.20"
replace Rate="2" if Rate=="0.02"



bys SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen Count=_n
bys SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen TotalCount=_N
keep if Count==1
drop Count TotalCount


gen TaxQuarter=0
replace TaxQuarter=17 if SourceFile=="t9854114"
replace TaxQuarter=18 if SourceFile=="t9854214"
replace TaxQuarter=19 if SourceFile=="t9854314"
replace TaxQuarter=20 if SourceFile=="t9854414"

bys DealerTIN TaxQuarter:gen TransActionCount=_N
by DealerTIN TaxQuarter:egen SumAmount=sum(Amount)
by DealerTIN TaxQuarter:egen SumTaxAmount=sum(TaxAmount)
by DealerTIN TaxQuarter :egen SumTotalAmount=sum(TotalAmount)
by DealerTIN TaxQuarter:gen Count=_n
keep if Count==1

drop Amount TaxAmount TotalAmount Rate Count

save "F:\2a2b_analysis\IntraFirmAnalysis\quarterly_dealers_2014.dta"

cd "E:\data"
use "annexure_2A2B_quarterly_2013.dta", clear
keep if SaleOrPurchase=="AE"

replace TransactionType="GD" if TransactionType=="GD "
replace TransactionType="GD" if TransactionType=="GD  "
replace TransactionType="GD" if TransactionType=="GD    "
replace TransactionType="GD" if TransactionType=="GD     "
replace TransactionType="GD" if TransactionType=="GD             "
replace TransactionType="GD" if TransactionType=="Gd"
replace TransactionType="GD" if TransactionType=="gd"
replace TransactionType="GD" if TransactionType=="gD"
replace TransactionType="GD" if TransactionType=="gd "
replace TransactionType="WC" if TransactionType=="wc"
replace TransactionType="WC" if TransactionType=="Wc"
replace TransactionType=trim(TransactionType)


drop T985DF5 T985DF6 T985DF7 T985DF8 FormsStatus T985DF4


replace Rate=trim(Rate)
replace Rate="0" if Rate=="0.0"
replace Rate="0" if Rate=="0.00"
replace Rate="0" if Rate=="00.00"
replace Rate="1" if Rate=="1.00"
replace Rate="1" if Rate=="1.000"
replace Rate="2" if Rate=="2.00"
replace Rate="12.5" if Rate=="12.50"
replace Rate="12.5" if Rate=="12.500"
replace Rate="12.5" if Rate=="12.5000"
replace Rate="20" if Rate=="20.00"
replace Rate="20" if Rate=="20.0"
replace Rate="5" if Rate=="5.0"
replace Rate="5" if Rate=="5.00"
replace Rate="5" if Rate=="5.000"
replace Rate="5" if Rate=="05.00"
replace Rate="4" if Rate=="4.00"
replace Rate="12.5" if Rate=="12.00"
replace Rate="12.5" if Rate=="0.12"
replace Rate="1" if Rate=="0.01"
replace Rate="5" if Rate=="0.05"
replace Rate="20" if Rate=="0.20"
replace Rate="2" if Rate=="0.02"


drop if Rate=="13.13"|Rate=="14"
drop if SellerBuyerTIN==""

bys SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen Count=_n
keep if Count==1
drop Count

gen TaxQuarter=0
replace TaxQuarter=13 if SourceFile=="t9854113"
replace TaxQuarter=14 if SourceFile=="t9854213"
replace TaxQuarter=15 if SourceFile=="t9854313"
replace TaxQuarter=16 if SourceFile=="t9854413"
 
bys DealerTIN TaxQuarter:gen TransActionCount=_N
by DealerTIN TaxQuarter:egen SumAmount=sum(Amount)
by DealerTIN TaxQuarter:egen SumTaxAmount=sum(TaxAmount)
by DealerTIN TaxQuarter :egen SumTotalAmount=sum(TotalAmount)
by DealerTIN TaxQuarter:gen Count=_n
keep if Count==1

drop Amount TaxAmount TotalAmount Rate Count

save "F:\2a2b_analysis\IntraFirmAnalysis\quarterly_dealers_2013.dta"

 
use "E:\data\annexure_2A2B_monthly_201213.dta", clear
keep if SaleOrPurchase=="AE"
replace SalePurchaseType="OT" if SalePurchaseType=="ot"
keep if SalePurchaseType=="CG"|SalePurchaseType=="OT"|SalePurchaseType==""

keep if DealerGoodType==""|DealerGoodType=="OT"|DealerGoodType=="CG"

drop if SellerBuyerTIN==""

replace Rate=trim(Rate)
replace Rate="0" if Rate=="0.0"
replace Rate="0" if Rate=="0.00"
replace Rate="0" if Rate=="00.00"
replace Rate="1" if Rate=="0.99"
replace Rate="1" if Rate=="1.00"
replace Rate="1" if Rate=="0.01"
replace Rate="1" if Rate=="1.000"
replace Rate="2" if Rate=="2.00"
replace Rate="2" if Rate=="0.02"
replace Rate="12.5" if Rate=="12.49"
replace Rate="12.5" if Rate=="12.00"
replace Rate="12.5" if Rate=="12.50"
replace Rate="12.5" if Rate=="12.500"
replace Rate="12.5" if Rate=="12.5000"
replace Rate="12.5" if Rate=="125.00"
replace Rate="12.5" if Rate==".12"
replace Rate="20" if Rate=="20.00"
replace Rate="20" if Rate=="20.0"
replace Rate="5" if Rate=="0.05"
replace Rate="5" if Rate==".05"
replace Rate="5" if Rate=="5.0"
replace Rate="5" if Rate=="5.00"
replace Rate="5" if Rate=="5.000"
replace Rate="5" if Rate=="5.0000"
replace Rate="5" if Rate=="05.00"
replace Rate="5" if Rate=="4.99"
replace Rate="4" if Rate=="4.00"
replace Rate="4" if Rate=="0.04"
replace Rate="10" if Rate=="0.10"
replace Rate="10" if Rate=="10.00"
replace Rate="12.5" if Rate=="0.12"


gen TaxQuarter=0
replace TaxQuarter=9 if SourceFile=="t985412"
replace TaxQuarter=9 if SourceFile=="t985512"
replace TaxQuarter=9 if SourceFile=="t985612"
replace TaxQuarter=10 if SourceFile=="t985712"
replace TaxQuarter=10 if SourceFile=="t985812"
replace TaxQuarter=10 if SourceFile=="t985912"
replace TaxQuarter=11 if SourceFile=="t9851012"
replace TaxQuarter=11 if SourceFile=="t9851112"
replace TaxQuarter=11 if SourceFile=="t9851212"
replace TaxQuarter=12 if SourceFile=="t985113"
replace TaxQuarter=12 if SourceFile=="t985213"
replace TaxQuarter=12 if SourceFile=="t985313"

gen dRate=Rate
destring dRate, replace
replace dRate=Amount if Rate!="5"&Rate!="4"&Rate!="12.5"&Rate!="1"&Rate!="0"&Rate!="2"&Rate!="20"&Rate!="0"&Rate!=""

gen d2Rate=Rate
destring d2Rate, replace
replace Amount=d2Rate if Rate!="5"&Rate!="4"&Rate!="12.5"&Rate!="1"&Rate!="0"&Rate!="2"&Rate!="20"&Rate!="0"&Rate!=""

tostring dRate, replace
replace Rate=dRate if Rate!="5"&Rate!="4"&Rate!="12.5"&Rate!="1"&Rate!="0"&Rate!="2"&Rate!="20"&Rate!="0"&Rate!=""

drop if Rate==""
drop dRate d2Rate

replace Rate="0" if Rate=="0.0"
replace Rate="0" if Rate=="0.00"
replace Rate="0" if Rate=="00.00"
replace Rate="1" if Rate=="0.99"
replace Rate="1" if Rate=="1.00"
replace Rate="1" if Rate=="0.01"
replace Rate="1" if Rate=="1.000"
replace Rate="2" if Rate=="2.00"
replace Rate="2" if Rate=="0.02"
replace Rate="12.5" if Rate=="12.49"
replace Rate="12.5" if Rate=="12.00"
replace Rate="12.5" if Rate=="12.50"
replace Rate="12.5" if Rate=="12.500"
replace Rate="12.5" if Rate=="12.5000"
replace Rate="12.5" if Rate=="125.00"
replace Rate="12.5" if Rate==".12"
replace Rate="20" if Rate=="20.00"
replace Rate="20" if Rate=="20.0"
replace Rate="5" if Rate=="0.05"
replace Rate="5" if Rate==".05"
replace Rate="5" if Rate=="5.0"
replace Rate="5" if Rate=="5.00"
replace Rate="5" if Rate=="5.000"
replace Rate="5" if Rate=="5.0000"
replace Rate="5" if Rate=="05.00"
replace Rate="5" if Rate=="4.99"
replace Rate="4" if Rate=="4.00"
replace Rate="4" if Rate=="0.04"
replace Rate="10" if Rate=="0.10"
replace Rate="10" if Rate=="10.00"
replace Rate="12.5" if Rate=="0.12"

keep if Rate=="5"|Rate=="4"|Rate=="12.5"|Rate=="1"|Rate=="0"|Rate=="2"|Rate=="20"|Rate=="0"

drop if DealerGoodType==""&SalePurchaseType==""

replace TransactionType="GD" if TransactionType=="GD "
replace TransactionType="GD" if TransactionType=="GD  "
replace TransactionType="GD" if TransactionType=="GD    "
replace TransactionType="GD" if TransactionType=="GD     "
replace TransactionType="GD" if TransactionType=="GD             "
replace TransactionType="GD" if TransactionType=="Gd"
replace TransactionType="GD" if TransactionType=="gd"
replace TransactionType="GD" if TransactionType=="gd "

replace TransactionType="WC" if TransactionType=="wc"
replace TransactionType="WC" if TransactionType=="Wc"

replace TransactionType=trim(TransactionType)

drop if TransactionType==""

bys SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen Count=_n
by SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen TotalCount=_N

keep if Count==1
drop Count TotalCount

bys DealerTIN Month:gen TransActionCount=_N
by DealerTIN Month:egen SumAmount=sum(Amount)
by DealerTIN Month:egen SumTaxAmount=sum(TaxAmount)
by DealerTIN Month:egen SumTotalAmount=sum(TotalAmount)
by DealerTIN Month:gen Count=_n

keep if Count==1
drop Count

save "F:\2a2b_analysis\IntraFirmAnalysis\monthly_dealers_2012.dta", replace

use "E:\data\PreliminaryAnalysis\returns\form16_data_v4_01122017.dta", clear
drop ReturnCount Time Tin T312203 T312202 Frequency T312020 T312021 DealerName DealerAddress DealerTelephone T312028 T312029 T312043 T312044 T312051 T312052 T312057 T312082 T312083 AccountNumber AccountType MICR BankDetails T312092 T312093 T312137 T312138 T312139 T312140 T312141 T312142 T312152 T312153 T312154 T312155 T312156 T312157 T312158 T312159 T312160 T312161 T312162 T312163 T312164 T312165 T312166 T312167
drop T312104 T312105 T312106 T312107 T312108 T312109 T312110 T312111 T312112 T312113 T312114 T312115 T312116 T312117 T312118 T312119 T312120 T312121 T312122 T312123 T312124 T312125 T312126 T312127 T312128 T312129 T312130 T312131 T312132 T312133 T312134 Name Designation Place Date TDSString Receipt2A2B T312170 T312DF1 T312DF2 T312DF3 T312DF4 T312DF5 T312DF6 T312DF7 T312DF8 Days T312178 Signatory
gen TaxQuarter=0
replace TaxQuarter=1 if TaxPeriod=="First Quarter-2010"
replace TaxQuarter=2 if TaxPeriod=="Second Quarter-2010"
replace TaxQuarter=3 if TaxPeriod=="Third Quarter-2010"
replace TaxQuarter=4 if TaxPeriod=="Fourth Quarter-2010"
replace TaxQuarter=5 if TaxPeriod=="First Quarter-2011"
replace TaxQuarter=6 if TaxPeriod=="Second Quarter-2011"
replace TaxQuarter=7 if TaxPeriod=="Third Quarter-2011"
replace TaxQuarter=8 if TaxPeriod=="Fourth Quarter-2011"
replace TaxQuarter=9 if TaxPeriod=="First Quarter-2012"
replace TaxQuarter=10 if TaxPeriod=="Second Quarter-2012"
replace TaxQuarter=11 if TaxPeriod=="Third Quarter-2012"
replace TaxQuarter=12 if TaxPeriod=="Fourth Quarter-2012"
replace TaxQuarter=13 if TaxPeriod=="First Quarter-2013"
replace TaxQuarter=14 if TaxPeriod=="Second Quarter-2013"
replace TaxQuarter=15 if TaxPeriod=="Third Quarter-2013"
replace TaxQuarter=16 if TaxPeriod=="Fourth Quarter-2013"
replace TaxQuarter=17 if TaxPeriod=="First Quarter-2014"
replace TaxQuarter=18 if TaxPeriod=="Second Quarter-2014"
replace TaxQuarter=19 if TaxPeriod=="Third Quarter-2014"
replace TaxQuarter=20 if TaxPeriod=="Fourth Quarter-2014"
gen TaxYear=0
replace TaxYear=1 if TaxQuarter>0&TaxQuarter<=4
replace TaxYear=2 if TaxQuarter>4&TaxQuarter<=8
replace TaxYear=3 if TaxQuarter>8&TaxQuarter<=12
replace TaxYear=4 if TaxQuarter>12&TaxQuarter<=16
replace TaxYear=5 if TaxQuarter>16&TaxQuarter<=20
gen TaxMonth=0
replace TaxMonth=1 if TaxPeriod=="Apr-2010"
replace TaxMonth=2 if TaxPeriod=="May-2010"
replace TaxMonth=3 if TaxPeriod=="Jun-2010"
replace TaxMonth=4 if TaxPeriod=="Jul-2010"
replace TaxMonth=5 if TaxPeriod=="Aug-2010"
replace TaxMonth=6 if TaxPeriod=="Sep-2010"
replace TaxMonth=7 if TaxPeriod=="Oct-2010"
replace TaxMonth=8 if TaxPeriod=="Nov-2010"
replace TaxMonth=9 if TaxPeriod=="Dec-2010"
replace TaxMonth=10 if TaxPeriod=="Jan-2011"
replace TaxMonth=11 if TaxPeriod=="Feb-2011"
replace TaxMonth=12 if TaxPeriod=="Mar-2011"
replace TaxMonth=13 if TaxPeriod=="Apr-2011"
replace TaxMonth=14 if TaxPeriod=="May-2011"
replace TaxMonth=15 if TaxPeriod=="Jun-2011"
replace TaxMonth=16 if TaxPeriod=="Jul-2011"
replace TaxMonth=17 if TaxPeriod=="Aug-2011"
replace TaxMonth=18 if TaxPeriod=="Sep-2011"
replace TaxMonth=19 if TaxPeriod=="Oct-2011"
replace TaxMonth=20 if TaxPeriod=="Nov-2011"
replace TaxMonth=21 if TaxPeriod=="Dec-2011"
replace TaxMonth=22 if TaxPeriod=="Jan-2012"
replace TaxMonth=23 if TaxPeriod=="Feb-2012"
replace TaxMonth=24 if TaxPeriod=="Mar-2012"
replace TaxMonth=25 if TaxPeriod=="Apr-2012"
replace TaxMonth=26 if TaxPeriod=="May-2012"
replace TaxMonth=27 if TaxPeriod=="Jun-2012"
replace TaxMonth=28 if TaxPeriod=="Jul-2012"
replace TaxMonth=29 if TaxPeriod=="Aug-2012"
replace TaxMonth=30 if TaxPeriod=="Sep-2012"
replace TaxMonth=31 if TaxPeriod=="Oct-2012"
replace TaxMonth=32 if TaxPeriod=="Nov-2012"
replace TaxMonth=33 if TaxPeriod=="Dec-2012"
replace TaxMonth=34 if TaxPeriod=="Jan-2013"
replace TaxMonth=35 if TaxPeriod=="Feb-2013"
replace TaxMonth=36 if TaxPeriod=="Mar-2013"
replace TaxMonth=37 if TaxPeriod=="Apr-2013"
replace TaxMonth=38 if TaxPeriod=="May-2013"
replace TaxYear=1 if TaxMonth>0&TaxMonth<=12
replace TaxYear=2 if TaxMonth>12&TaxMonth<=24
replace TaxYear=3 if TaxMonth>24&TaxMonth<=36
replace TaxYear=4 if TaxMonth>36
replace TaxQuarter=1 if TaxMonth>0&TaxMonth<=3
replace TaxQuarter=2 if TaxMonth>3&TaxMonth<=6
replace TaxQuarter=3 if TaxMonth>6&TaxMonth<=9
replace TaxQuarter=4 if TaxMonth>9&TaxMonth<=12
replace TaxQuarter=5 if TaxMonth>12&TaxMonth<=15
replace TaxQuarter=6 if TaxMonth>15&TaxMonth<=18
replace TaxQuarter=7 if TaxMonth>18&TaxMonth<=21
replace TaxQuarter=8 if TaxMonth>21&TaxMonth<=24
replace TaxQuarter=9 if TaxMonth>24&TaxMonth<=27
replace TaxQuarter=10 if TaxMonth>27&TaxMonth<=30
replace TaxQuarter=11 if TaxMonth>30&TaxMonth<=33
replace TaxQuarter=12 if TaxMonth>33&TaxMonth<=36
replace TaxQuarter=13 if TaxMonth>36&TaxMonth<=39
label define year 1 "2010-11" 2 "2011-12" 3 "2012-13" 4 "2013-14" 5 "2014-15" 6 "2015-16"
label define quarter 1 "Q1, 2010-11" 2 "Q2, 2010-11" 3 "Q3, 2010-11" 4 "Q4, 2010-11" 5 "Q1, 2011-12" 6 "Q2, 2011-12" 7 "Q3, 2011-12" 8 "Q4, 2011-12" 9 "Q1, 2012-13" 10 "Q2, 2012-13" 11 "Q3, 2012-13" 12 "Q4, 2012-13" 13 "Q1, 2013-14" 14 "Q2, 2013-14" 15 "Q3, 2013-14" 16 "Q4, 2013-14" 17 "Q1, 2014-15" 18 "Q2, 2014-15" 19 "Q3, 2014-15" 20 "Q4, 2014-15"
label values TaxQuarter quarter
label values TaxYear year

keep if TaxQuarter==9|TaxQuarter==10|TaxQuarter==11|TaxQuarter==12
gen Month=0
replace Month=4 if TaxMonth==25
replace Month=5 if TaxMonth==26
replace Month=6 if TaxMonth==27
replace Month=7 if TaxMonth==28
replace Month=8 if TaxMonth==29
replace Month=9 if TaxMonth==30
replace Month=10 if TaxMonth==31
replace Month=11 if TaxMonth==32
replace Month=12 if TaxMonth==33
replace Month=1 if TaxMonth==34
replace Month=2 if TaxMonth==35
replace Month=3 if TaxMonth==36

gen TypeDealer="Quarterly" if TaxMonth==0
replace TypeDealer="Monthly" if TaxMonth!=0

keep DealerTIN TaxQuarter TaxMonth TypeDealer Month

save "F:\2a2b_analysis\IntraFirmAnalysis\DealerType.dta", replace


use "F:\2a2b_analysis\IntraFirmAnalysis\monthly_dealers_2012.dta", clear
merge m:m DealerTIN TaxQuarter Month using "F:\2a2b_analysis\IntraFirmAnalysis\DealerType.dta", keepusing(TypeDealer TaxQuarter TaxMonth)
keep if TypeDealer=="Monthly"&_merge==3
drop Amount TaxAmount TotalAmount Rate
save "F:\2a2b_analysis\IntraFirmAnalysis\monthly_2a2b_combined.dta"


use "F:\2a2b_analysis\IntraFirmAnalysis\monthly_dealers_2012.dta", clear
merge m:m DealerTIN TaxQuarter using "F:\2a2b_analysis\IntraFirmAnalysis\DealerType.dta", keepusing(TypeDealer TaxQuarter TaxMonth)
keep if TypeDealer=="Quarterly"&_merge==3
drop Amount TaxAmount TotalAmount Rate
rename TransActionCount ActionCount
rename SumAmount Amount
rename SumTaxAmount TaxAmount
rename SumTotalAmount TotalAmount
bys DealerTIN TaxQuarter:gen TransActionCount=sum(ActionCount)
by DealerTIN TaxQuarter:egen SumAmount=sum(Amount)
by DealerTIN TaxQuarter:egen SumTaxAmount=sum(TaxAmount)
by DealerTIN TaxQuarter:egen SumTotalAmount=sum(TotalAmount)
by DealerTIN TaxQuarter:gen Count=_n
by DealerTIN TaxQuarter: gen TotalCount=_N
keep if Count==1
drop Amount TaxAmount TotalAmount Count
save "F:\2a2b_analysis\IntraFirmAnalysis\quarterly_2a2b_combined.dta", replace

use "F:\2a2b_analysis\IntraFirmAnalysis\monthly_2a2b_combined.dta", clear
append using "F:\2a2b_analysis\IntraFirmAnalysis\quarterly_2a2b_combined.dta"
save "F:\2a2b_analysis\IntraFirmAnalysis\Combined2a2b.dta", replace

use "E:\data\PreliminaryAnalysis\returns\form16_data_v4_01122017.dta", clear
drop  ReturnCount Time Tin T312203 T312202 Frequency T312020 T312021 DealerName DealerAddress DealerTelephone T312028 T312029 T312043 T312044 T312051 T312052 T312057 T312082 T312083 AccountNumber AccountType MICR BankDetails T312092 T312093 T312137 T312138 T312139 T312140 T312141 T312142 T312152 T312153 T312154 T312155 T312156 T312157 T312158 T312159 T312160 T312161 T312162 T312163 T312164 T312165 T312166 T312167
drop T312104 T312105 T312106 T312107 T312108 T312109 T312110 T312111 T312112 T312113 T312114 T312115 T312116 T312117 T312118 T312119 T312120 T312121 T312122 T312123 T312124 T312125 T312126 T312127 T312128 T312129 T312130 T312131 T312132 T312133 T312134 Name Designation Place Date TDSString Receipt2A2B T312170 T312DF1 T312DF2 T312DF3 T312DF4 T312DF5 T312DF6 T312DF7 T312DF8 Days T312178 Signatory
gen TaxQuarter=0
replace TaxQuarter=1 if TaxPeriod=="First Quarter-2010"
replace TaxQuarter=2 if TaxPeriod=="Second Quarter-2010"
replace TaxQuarter=3 if TaxPeriod=="Third Quarter-2010"
replace TaxQuarter=4 if TaxPeriod=="Fourth Quarter-2010"
replace TaxQuarter=5 if TaxPeriod=="First Quarter-2011"
replace TaxQuarter=6 if TaxPeriod=="Second Quarter-2011"
replace TaxQuarter=7 if TaxPeriod=="Third Quarter-2011"
replace TaxQuarter=8 if TaxPeriod=="Fourth Quarter-2011"
replace TaxQuarter=9 if TaxPeriod=="First Quarter-2012"
replace TaxQuarter=10 if TaxPeriod=="Second Quarter-2012"
replace TaxQuarter=11 if TaxPeriod=="Third Quarter-2012"
replace TaxQuarter=12 if TaxPeriod=="Fourth Quarter-2012"
replace TaxQuarter=13 if TaxPeriod=="First Quarter-2013"
replace TaxQuarter=14 if TaxPeriod=="Second Quarter-2013"
replace TaxQuarter=15 if TaxPeriod=="Third Quarter-2013"
replace TaxQuarter=16 if TaxPeriod=="Fourth Quarter-2013"
replace TaxQuarter=17 if TaxPeriod=="First Quarter-2014"
replace TaxQuarter=18 if TaxPeriod=="Second Quarter-2014"
replace TaxQuarter=19 if TaxPeriod=="Third Quarter-2014"
replace TaxQuarter=20 if TaxPeriod=="Fourth Quarter-2014"
gen TaxYear=0
replace TaxYear=1 if TaxQuarter>0&TaxQuarter<=4
replace TaxYear=2 if TaxQuarter>4&TaxQuarter<=8
replace TaxYear=3 if TaxQuarter>8&TaxQuarter<=12
replace TaxYear=4 if TaxQuarter>12&TaxQuarter<=16
replace TaxYear=5 if TaxQuarter>16&TaxQuarter<=20
gen TaxMonth=0
replace TaxMonth=1 if TaxPeriod=="Apr-2010"
replace TaxMonth=2 if TaxPeriod=="May-2010"
replace TaxMonth=3 if TaxPeriod=="Jun-2010"
replace TaxMonth=4 if TaxPeriod=="Jul-2010"
replace TaxMonth=5 if TaxPeriod=="Aug-2010"
replace TaxMonth=6 if TaxPeriod=="Sep-2010"
replace TaxMonth=7 if TaxPeriod=="Oct-2010"
replace TaxMonth=8 if TaxPeriod=="Nov-2010"
replace TaxMonth=9 if TaxPeriod=="Dec-2010"
replace TaxMonth=10 if TaxPeriod=="Jan-2011"
replace TaxMonth=11 if TaxPeriod=="Feb-2011"
replace TaxMonth=12 if TaxPeriod=="Mar-2011"
replace TaxMonth=13 if TaxPeriod=="Apr-2011"
replace TaxMonth=14 if TaxPeriod=="May-2011"
replace TaxMonth=15 if TaxPeriod=="Jun-2011"
replace TaxMonth=16 if TaxPeriod=="Jul-2011"
replace TaxMonth=17 if TaxPeriod=="Aug-2011"
replace TaxMonth=18 if TaxPeriod=="Sep-2011"
replace TaxMonth=19 if TaxPeriod=="Oct-2011"
replace TaxMonth=20 if TaxPeriod=="Nov-2011"
replace TaxMonth=21 if TaxPeriod=="Dec-2011"
replace TaxMonth=22 if TaxPeriod=="Jan-2012"
replace TaxMonth=23 if TaxPeriod=="Feb-2012"
replace TaxMonth=24 if TaxPeriod=="Mar-2012"
replace TaxMonth=25 if TaxPeriod=="Apr-2012"
replace TaxMonth=26 if TaxPeriod=="May-2012"
replace TaxMonth=27 if TaxPeriod=="Jun-2012"
replace TaxMonth=28 if TaxPeriod=="Jul-2012"
replace TaxMonth=29 if TaxPeriod=="Aug-2012"
replace TaxMonth=30 if TaxPeriod=="Sep-2012"
replace TaxMonth=31 if TaxPeriod=="Oct-2012"
replace TaxMonth=32 if TaxPeriod=="Nov-2012"
replace TaxMonth=33 if TaxPeriod=="Dec-2012"
replace TaxMonth=34 if TaxPeriod=="Jan-2013"
replace TaxMonth=35 if TaxPeriod=="Feb-2013"
replace TaxMonth=36 if TaxPeriod=="Mar-2013"
replace TaxMonth=37 if TaxPeriod=="Apr-2013"
replace TaxMonth=38 if TaxPeriod=="May-2013"
replace TaxYear=1 if TaxMonth>0&TaxMonth<=12
replace TaxYear=2 if TaxMonth>12&TaxMonth<=24
replace TaxYear=3 if TaxMonth>24&TaxMonth<=36
replace TaxYear=4 if TaxMonth>36
replace TaxQuarter=1 if TaxMonth>0&TaxMonth<=3
replace TaxQuarter=2 if TaxMonth>3&TaxMonth<=6
replace TaxQuarter=3 if TaxMonth>6&TaxMonth<=9
replace TaxQuarter=4 if TaxMonth>9&TaxMonth<=12
replace TaxQuarter=5 if TaxMonth>12&TaxMonth<=15
replace TaxQuarter=6 if TaxMonth>15&TaxMonth<=18
replace TaxQuarter=7 if TaxMonth>18&TaxMonth<=21
replace TaxQuarter=8 if TaxMonth>21&TaxMonth<=24
replace TaxQuarter=9 if TaxMonth>24&TaxMonth<=27
replace TaxQuarter=10 if TaxMonth>27&TaxMonth<=30
replace TaxQuarter=11 if TaxMonth>30&TaxMonth<=33
replace TaxQuarter=12 if TaxMonth>33&TaxMonth<=36
replace TaxQuarter=13 if TaxMonth>36&TaxMonth<=39
label define year 1 "2010-11" 2 "2011-12" 3 "2012-13" 4 "2013-14" 5 "2014-15" 6 "2015-16"
label define quarter 1 "Q1, 2010-11" 2 "Q2, 2010-11" 3 "Q3, 2010-11" 4 "Q4, 2010-11" 5 "Q1, 2011-12" 6 "Q2, 2011-12" 7 "Q3, 2011-12" 8 "Q4, 2011-12" 9 "Q1, 2012-13" 10 "Q2, 2012-13" 11 "Q3, 2012-13" 12 "Q4, 2012-13" 13 "Q1, 2013-14" 14 "Q2, 2013-14" 15 "Q3, 2013-14" 16 "Q4, 2013-14" 17 "Q1, 2014-15" 18 "Q2, 2014-15" 19 "Q3, 2014-15" 20 "Q4, 2014-15"
label values TaxQuarter quarter
label values TaxYear year


drop if TaxQuarter==0
//keep if TaxQuarter==9|TaxQuarter==10|TaxQuarter==11|TaxQuarter==12
gen Month=0
replace Month=4 if TaxMonth==25
replace Month=5 if TaxMonth==26
replace Month=6 if TaxMonth==27
replace Month=7 if TaxMonth==28
replace Month=8 if TaxMonth==29
replace Month=9 if TaxMonth==30
replace Month=10 if TaxMonth==31
replace Month=11 if TaxMonth==32
replace Month=12 if TaxMonth==33
replace Month=1 if TaxMonth==34
replace Month=2 if TaxMonth==35
replace Month=3 if TaxMonth==36

gen TypeDealer="Quarterly" if TaxMonth==0
replace TypeDealer="Monthly" if TaxMonth!=0

gen MoneyDeposited=max(AggregateAmountPaid, AmountDepositedByDealer)
gen PositiveContribution=0
replace PositiveContribution=1 if MoneyDeposited>0

merge 1:1 DealerTIN  TypeDealer TaxMonth TaxQuarter using "F:\2a2b_analysis\IntraFirmAnalysis\Combined2a2b.dta", keepusing(TransActionCount SumAmount SumTaxAmount SumTotalAmount) generate(_merge_combine)

rename TransActionCount TransActionCount1 
rename SumAmount SumAmount1 
rename SumTaxAmount SumTaxAmount1
rename SumTotalAmount SumTotalAmount1

gen PurchaseRatio=TaxCreditBeforeAdjustment/SumTaxAmount


use "E:\data\PreliminaryAnalysis\returns\form16_data_v4_01122017.dta", clear
drop groupid ReturnCount Time Tin T312203 T312202 Frequency T312020 T312021 DealerName DealerAddress DealerTelephone T312028 T312029 T312043 T312044 T312051 T312052 T312057 T312082 T312083 AccountNumber AccountType MICR BankDetails T312092 T312093 T312137 T312138 T312139 T312140 T312141 T312142 T312152 T312153 T312154 T312155 T312156 T312157 T312158 T312159 T312160 T312161 T312162 T312163 T312164 T312165 T312166 T312167
drop T312104 T312105 T312106 T312107 T312108 T312109 T312110 T312111 T312112 T312113 T312114 T312115 T312116 T312117 T312118 T312119 T312120 T312121 T312122 T312123 T312124 T312125 T312126 T312127 T312128 T312129 T312130 T312131 T312132 T312133 T312134 Name Designation Place Date TDSString Receipt2A2B T312170 T312DF1 T312DF2 T312DF3 T312DF4 T312DF5 T312DF6 T312DF7 T312DF8 Days T312178 Signatory
gen TaxQuarter=0
replace TaxQuarter=1 if TaxPeriod=="First Quarter-2010"
replace TaxQuarter=2 if TaxPeriod=="Second Quarter-2010"
replace TaxQuarter=3 if TaxPeriod=="Third Quarter-2010"
replace TaxQuarter=4 if TaxPeriod=="Fourth Quarter-2010"
replace TaxQuarter=5 if TaxPeriod=="First Quarter-2011"
replace TaxQuarter=6 if TaxPeriod=="Second Quarter-2011"
replace TaxQuarter=7 if TaxPeriod=="Third Quarter-2011"
replace TaxQuarter=8 if TaxPeriod=="Fourth Quarter-2011"
replace TaxQuarter=9 if TaxPeriod=="First Quarter-2012"
replace TaxQuarter=10 if TaxPeriod=="Second Quarter-2012"
replace TaxQuarter=11 if TaxPeriod=="Third Quarter-2012"
replace TaxQuarter=12 if TaxPeriod=="Fourth Quarter-2012"
replace TaxQuarter=13 if TaxPeriod=="First Quarter-2013"
replace TaxQuarter=14 if TaxPeriod=="Second Quarter-2013"
replace TaxQuarter=15 if TaxPeriod=="Third Quarter-2013"
replace TaxQuarter=16 if TaxPeriod=="Fourth Quarter-2013"
replace TaxQuarter=17 if TaxPeriod=="First Quarter-2014"
replace TaxQuarter=18 if TaxPeriod=="Second Quarter-2014"
replace TaxQuarter=19 if TaxPeriod=="Third Quarter-2014"
replace TaxQuarter=20 if TaxPeriod=="Fourth Quarter-2014"
gen TaxYear=0
replace TaxYear=1 if TaxQuarter>0&TaxQuarter<=4
replace TaxYear=2 if TaxQuarter>4&TaxQuarter<=8
replace TaxYear=3 if TaxQuarter>8&TaxQuarter<=12
replace TaxYear=4 if TaxQuarter>12&TaxQuarter<=16
replace TaxYear=5 if TaxQuarter>16&TaxQuarter<=20
gen TaxMonth=0
replace TaxMonth=1 if TaxPeriod=="Apr-2010"
replace TaxMonth=2 if TaxPeriod=="May-2010"
replace TaxMonth=3 if TaxPeriod=="Jun-2010"
replace TaxMonth=4 if TaxPeriod=="Jul-2010"
replace TaxMonth=5 if TaxPeriod=="Aug-2010"
replace TaxMonth=6 if TaxPeriod=="Sep-2010"
replace TaxMonth=7 if TaxPeriod=="Oct-2010"
replace TaxMonth=8 if TaxPeriod=="Nov-2010"
replace TaxMonth=9 if TaxPeriod=="Dec-2010"
replace TaxMonth=10 if TaxPeriod=="Jan-2011"
replace TaxMonth=11 if TaxPeriod=="Feb-2011"
replace TaxMonth=12 if TaxPeriod=="Mar-2011"
replace TaxMonth=13 if TaxPeriod=="Apr-2011"
replace TaxMonth=14 if TaxPeriod=="May-2011"
replace TaxMonth=15 if TaxPeriod=="Jun-2011"
replace TaxMonth=16 if TaxPeriod=="Jul-2011"
replace TaxMonth=17 if TaxPeriod=="Aug-2011"
replace TaxMonth=18 if TaxPeriod=="Sep-2011"
replace TaxMonth=19 if TaxPeriod=="Oct-2011"
replace TaxMonth=20 if TaxPeriod=="Nov-2011"
replace TaxMonth=21 if TaxPeriod=="Dec-2011"
replace TaxMonth=22 if TaxPeriod=="Jan-2012"
replace TaxMonth=23 if TaxPeriod=="Feb-2012"
replace TaxMonth=24 if TaxPeriod=="Mar-2012"
replace TaxMonth=25 if TaxPeriod=="Apr-2012"
replace TaxMonth=26 if TaxPeriod=="May-2012"
replace TaxMonth=27 if TaxPeriod=="Jun-2012"
replace TaxMonth=28 if TaxPeriod=="Jul-2012"
replace TaxMonth=29 if TaxPeriod=="Aug-2012"
replace TaxMonth=30 if TaxPeriod=="Sep-2012"
replace TaxMonth=31 if TaxPeriod=="Oct-2012"
replace TaxMonth=32 if TaxPeriod=="Nov-2012"
replace TaxMonth=33 if TaxPeriod=="Dec-2012"
replace TaxMonth=34 if TaxPeriod=="Jan-2013"
replace TaxMonth=35 if TaxPeriod=="Feb-2013"
replace TaxMonth=36 if TaxPeriod=="Mar-2013"
replace TaxMonth=37 if TaxPeriod=="Apr-2013"
replace TaxMonth=38 if TaxPeriod=="May-2013"
replace TaxYear=1 if TaxMonth>0&TaxMonth<=12
replace TaxYear=2 if TaxMonth>12&TaxMonth<=24
replace TaxYear=3 if TaxMonth>24&TaxMonth<=36
replace TaxYear=4 if TaxMonth>36
replace TaxQuarter=1 if TaxMonth>0&TaxMonth<=3
replace TaxQuarter=2 if TaxMonth>3&TaxMonth<=6
replace TaxQuarter=3 if TaxMonth>6&TaxMonth<=9
replace TaxQuarter=4 if TaxMonth>9&TaxMonth<=12
replace TaxQuarter=5 if TaxMonth>12&TaxMonth<=15
replace TaxQuarter=6 if TaxMonth>15&TaxMonth<=18
replace TaxQuarter=7 if TaxMonth>18&TaxMonth<=21
replace TaxQuarter=8 if TaxMonth>21&TaxMonth<=24
replace TaxQuarter=9 if TaxMonth>24&TaxMonth<=27
replace TaxQuarter=10 if TaxMonth>27&TaxMonth<=30
replace TaxQuarter=11 if TaxMonth>30&TaxMonth<=33
replace TaxQuarter=12 if TaxMonth>33&TaxMonth<=36
replace TaxQuarter=13 if TaxMonth>36&TaxMonth<=39
label define year 1 "2010-11" 2 "2011-12" 3 "2012-13" 4 "2013-14" 5 "2014-15" 6 "2015-16"
label define quarter 1 "Q1, 2010-11" 2 "Q2, 2010-11" 3 "Q3, 2010-11" 4 "Q4, 2010-11" 5 "Q1, 2011-12" 6 "Q2, 2011-12" 7 "Q3, 2011-12" 8 "Q4, 2011-12" 9 "Q1, 2012-13" 10 "Q2, 2012-13" 11 "Q3, 2012-13" 12 "Q4, 2012-13" 13 "Q1, 2013-14" 14 "Q2, 2013-14" 15 "Q3, 2013-14" 16 "Q4, 2013-14" 17 "Q1, 2014-15" 18 "Q2, 2014-15" 19 "Q3, 2014-15" 20 "Q4, 2014-15"
label values TaxQuarter quarter
label values TaxYear year

keep if TaxQuarter==13|TaxQuarter==14|TaxQuarter==15|TaxQuarter==16
drop if TaxPeriod=="Apr-2013"|TaxPeriod=="May-2013"

//merge 1:1 DealerTIN TaxQuarter using "Z:\quarterly_dealers_2013.dta", keepusing(TransActionCount SumAmount SumTaxAmount SumTotalAmount) generate(_merge_combine2)
merge m:1 DealerTIN TaxQuarter using "F:\2a2b_analysis\IntraFirmAnalysis\quarterly_dealers_2013.dta", keepusing(TransActionCount SumAmount SumTaxAmount SumTotalAmount) generate(_merge_combine2)
replace TransActionCount=TransActionCount1 if TransActionCount==.
replace SumAmount=SumAmount1 if SumAmount==.
replace SumTaxAmount=SumTaxAmount1 if SumTaxAmount==.
replace SumTotalAmount=SumTotalAmount1 if SumTotalAmount==.

rename TransActionCount TransActionCount2 
rename SumAmount SumAmount2
rename SumTaxAmount SumTaxAmount2
rename SumTotalAmount SumTotalAmount2

gen PurchaseRatio=TaxCreditBeforeAdjustment/SumTaxAmount



use "D:\data\PreliminaryAnalysis\returns\form16_data_v3_0901.dta" , clear
drop groupid ReturnCount Time Tin T312203 T312202 Frequency T312020 T312021 DealerName DealerAddress DealerTelephone T312028 T312029 T312043 T312044 T312051 T312052 T312057 T312082 T312083 AccountNumber AccountType MICR BankDetails T312092 T312093 T312137 T312138 T312139 T312140 T312141 T312142 T312152 T312153 T312154 T312155 T312156 T312157 T312158 T312159 T312160 T312161 T312162 T312163 T312164 T312165 T312166 T312167
drop T312104 T312105 T312106 T312107 T312108 T312109 T312110 T312111 T312112 T312113 T312114 T312115 T312116 T312117 T312118 T312119 T312120 T312121 T312122 T312123 T312124 T312125 T312126 T312127 T312128 T312129 T312130 T312131 T312132 T312133 T312134 Name Designation Place Date TDSString Receipt2A2B T312170 T312DF1 T312DF2 T312DF3 T312DF4 T312DF5 T312DF6 T312DF7 T312DF8 Days T312178 Signatory
gen TaxQuarter=0
replace TaxQuarter=1 if TaxPeriod=="First Quarter-2010"
replace TaxQuarter=2 if TaxPeriod=="Second Quarter-2010"
replace TaxQuarter=3 if TaxPeriod=="Third Quarter-2010"
replace TaxQuarter=4 if TaxPeriod=="Fourth Quarter-2010"
replace TaxQuarter=5 if TaxPeriod=="First Quarter-2011"
replace TaxQuarter=6 if TaxPeriod=="Second Quarter-2011"
replace TaxQuarter=7 if TaxPeriod=="Third Quarter-2011"
replace TaxQuarter=8 if TaxPeriod=="Fourth Quarter-2011"
replace TaxQuarter=9 if TaxPeriod=="First Quarter-2012"
replace TaxQuarter=10 if TaxPeriod=="Second Quarter-2012"
replace TaxQuarter=11 if TaxPeriod=="Third Quarter-2012"
replace TaxQuarter=12 if TaxPeriod=="Fourth Quarter-2012"
replace TaxQuarter=13 if TaxPeriod=="First Quarter-2013"
replace TaxQuarter=14 if TaxPeriod=="Second Quarter-2013"
replace TaxQuarter=15 if TaxPeriod=="Third Quarter-2013"
replace TaxQuarter=16 if TaxPeriod=="Fourth Quarter-2013"
replace TaxQuarter=17 if TaxPeriod=="First Quarter-2014"
replace TaxQuarter=18 if TaxPeriod=="Second Quarter-2014"
replace TaxQuarter=19 if TaxPeriod=="Third Quarter-2014"
replace TaxQuarter=20 if TaxPeriod=="Fourth Quarter-2014"
gen TaxYear=0
replace TaxYear=1 if TaxQuarter>0&TaxQuarter<=4
replace TaxYear=2 if TaxQuarter>4&TaxQuarter<=8
replace TaxYear=3 if TaxQuarter>8&TaxQuarter<=12
replace TaxYear=4 if TaxQuarter>12&TaxQuarter<=16
replace TaxYear=5 if TaxQuarter>16&TaxQuarter<=20
gen TaxMonth=0
replace TaxMonth=1 if TaxPeriod=="Apr-2010"
replace TaxMonth=2 if TaxPeriod=="May-2010"
replace TaxMonth=3 if TaxPeriod=="Jun-2010"
replace TaxMonth=4 if TaxPeriod=="Jul-2010"
replace TaxMonth=5 if TaxPeriod=="Aug-2010"
replace TaxMonth=6 if TaxPeriod=="Sep-2010"
replace TaxMonth=7 if TaxPeriod=="Oct-2010"
replace TaxMonth=8 if TaxPeriod=="Nov-2010"
replace TaxMonth=9 if TaxPeriod=="Dec-2010"
replace TaxMonth=10 if TaxPeriod=="Jan-2011"
replace TaxMonth=11 if TaxPeriod=="Feb-2011"
replace TaxMonth=12 if TaxPeriod=="Mar-2011"
replace TaxMonth=13 if TaxPeriod=="Apr-2011"
replace TaxMonth=14 if TaxPeriod=="May-2011"
replace TaxMonth=15 if TaxPeriod=="Jun-2011"
replace TaxMonth=16 if TaxPeriod=="Jul-2011"
replace TaxMonth=17 if TaxPeriod=="Aug-2011"
replace TaxMonth=18 if TaxPeriod=="Sep-2011"
replace TaxMonth=19 if TaxPeriod=="Oct-2011"
replace TaxMonth=20 if TaxPeriod=="Nov-2011"
replace TaxMonth=21 if TaxPeriod=="Dec-2011"
replace TaxMonth=22 if TaxPeriod=="Jan-2012"
replace TaxMonth=23 if TaxPeriod=="Feb-2012"
replace TaxMonth=24 if TaxPeriod=="Mar-2012"
replace TaxMonth=25 if TaxPeriod=="Apr-2012"
replace TaxMonth=26 if TaxPeriod=="May-2012"
replace TaxMonth=27 if TaxPeriod=="Jun-2012"
replace TaxMonth=28 if TaxPeriod=="Jul-2012"
replace TaxMonth=29 if TaxPeriod=="Aug-2012"
replace TaxMonth=30 if TaxPeriod=="Sep-2012"
replace TaxMonth=31 if TaxPeriod=="Oct-2012"
replace TaxMonth=32 if TaxPeriod=="Nov-2012"
replace TaxMonth=33 if TaxPeriod=="Dec-2012"
replace TaxMonth=34 if TaxPeriod=="Jan-2013"
replace TaxMonth=35 if TaxPeriod=="Feb-2013"
replace TaxMonth=36 if TaxPeriod=="Mar-2013"
replace TaxMonth=37 if TaxPeriod=="Apr-2013"
replace TaxMonth=38 if TaxPeriod=="May-2013"
replace TaxYear=1 if TaxMonth>0&TaxMonth<=12
replace TaxYear=2 if TaxMonth>12&TaxMonth<=24
replace TaxYear=3 if TaxMonth>24&TaxMonth<=36
replace TaxYear=4 if TaxMonth>36
replace TaxQuarter=1 if TaxMonth>0&TaxMonth<=3
replace TaxQuarter=2 if TaxMonth>3&TaxMonth<=6
replace TaxQuarter=3 if TaxMonth>6&TaxMonth<=9
replace TaxQuarter=4 if TaxMonth>9&TaxMonth<=12
replace TaxQuarter=5 if TaxMonth>12&TaxMonth<=15
replace TaxQuarter=6 if TaxMonth>15&TaxMonth<=18
replace TaxQuarter=7 if TaxMonth>18&TaxMonth<=21
replace TaxQuarter=8 if TaxMonth>21&TaxMonth<=24
replace TaxQuarter=9 if TaxMonth>24&TaxMonth<=27
replace TaxQuarter=10 if TaxMonth>27&TaxMonth<=30
replace TaxQuarter=11 if TaxMonth>30&TaxMonth<=33
replace TaxQuarter=12 if TaxMonth>33&TaxMonth<=36
replace TaxQuarter=13 if TaxMonth>36&TaxMonth<=39
label define year 1 "2010-11" 2 "2011-12" 3 "2012-13" 4 "2013-14" 5 "2014-15" 6 "2015-16"
label define quarter 1 "Q1, 2010-11" 2 "Q2, 2010-11" 3 "Q3, 2010-11" 4 "Q4, 2010-11" 5 "Q1, 2011-12" 6 "Q2, 2011-12" 7 "Q3, 2011-12" 8 "Q4, 2011-12" 9 "Q1, 2012-13" 10 "Q2, 2012-13" 11 "Q3, 2012-13" 12 "Q4, 2012-13" 13 "Q1, 2013-14" 14 "Q2, 2013-14" 15 "Q3, 2013-14" 16 "Q4, 2013-14" 17 "Q1, 2014-15" 18 "Q2, 2014-15" 19 "Q3, 2014-15" 20 "Q4, 2014-15"
label values TaxQuarter quarter
label values TaxYear year

keep if TaxQuarter==17|TaxQuarter==18|TaxQuarter==19|TaxQuarter==20

//merge 1:1 DealerTIN TaxQuarter using "Z:\quarterly_dealers_2014.dta", keepusing(TransActionCount SumAmount SumTaxAmount SumTotalAmount) generate(_merge_combine3)
merge m:1 DealerTIN TaxQuarter using "F:\2a2b_analysis\IntraFirmAnalysis\quarterly_dealers_2014.dta", keepusing(TransActionCount SumAmount SumTaxAmount SumTotalAmount) generate(_merge_combine3)

rename TransActionCount TransActionCount1 
rename SumAmount SumAmount1 
rename SumTaxAmount SumTaxAmount1
rename SumTotalAmount SumTotalAmount1

replace TransActionCount=TransActionCount2 if TransActionCount==.
replace SumAmount=SumAmount2 if SumAmount==.
replace SumTaxAmount=SumTaxAmount2 if SumTaxAmount==.
replace SumTotalAmount=SumTotalAmount2 if SumTotalAmount==.


gen PurchaseRatio=TaxCreditBeforeAdjustment/SumTaxAmount
gen OtherPurchaseRatio=CreditOtherGoods/SumTaxAmount


preserve
collapse (mean) PurchaseRatio, by(DealerTIN TaxQuarter)
collapse (mean) PurchaseRatio, by(TaxQuarter)
twoway (connected PurchaseRatio TaxQuarter if PurchaseRatio!=.)
restore 

merge m:1 DealerTIN using "E:\data\PreliminaryAnalysis\BogusDealers\BogusIdentifiedFromOnlineGovernment.dta", generate(_merge_bogus2)
drop if _merge_bogus2==2
gen SecondBogusDummy=0
replace SecondBogusDummy=1 if Bogus=="YES"

preserve
collapse (mean) PurchaseRatio, by(TaxQuarter SecondBogusDummy)
twoway (connected PurchaseRatio TaxQuarter if PurchaseRatio!=.&SecondBogusDummy==0) (connected PurchaseRatio TaxQuarter if PurchaseRatio!=.&SecondBogusDummy==1)
restore 


sum ratio if TaxMonth!=0
sum ratio if TaxMonth==0
sum TurnoverGross TurnoverLocal MoneyDeposited PositiveContribution if _merge_combine==1&TypeDealer=="Quarterly", detail
sum TurnoverGross TurnoverLocal MoneyDeposited PositiveContribution if _merge_combine==3&TypeDealer=="Quarterly", detail

sum TurnoverGross TurnoverLocal MoneyDeposited PositiveContribution if _merge_combine==1&TypeDealer=="Monthly", detail
sum TurnoverGross TurnoverLocal MoneyDeposited PositiveContribution if _merge_combine==3&TypeDealer=="Monthly", detail


use "Z:\intrapurchase_all.dta", clear
keep if TaxMonth==0
gsort DealerTIN TaxQuarter
by DealerTIN: gen TotalCount=_N
keep if TotalCount==20
tab TaxQuarter
tab _merge_combine
tab _merge_combine2
tab _merge_combine3
gen Dummy=0
replace Dummy=1 if PurchaseRatio!=.
gsort DealerTIN Dummy
by DealerTIN: replace Dummy=Dummy[_n-1] if Dummy>=.
br
replace Dummy=. if Dummy==0
gsort DealerTIN Dummy
by DealerTIN: replace Dummy=Dummy[_n-1] if Dummy>=.
preserve
collapse (mean) TaxCreditBeforeAdjustment SumTaxAmount, by(TaxQuarter Dummy)
twoway (connected TaxCreditBeforeAdjustment TaxQuarter) (connected SumTaxAmount TaxQuarter) if Dummy==1



merge

keep if TaxQuarter==9
tab TaxPeriod
isid DealerTIN
bys DealerTIN: gen Count=_N
tab Count
keep if TaxPeriod=="Apr-2012"|TaxPeriod=="First Quarter-2012"
isid DealerTIN
drop Count
bys DealerTIN: gen Count=_N
tab Count
bys DealerTIN: gen RunningCount=_n
keep if RunningCount==1
merge 1:1 DealerTIN using "Z:\monthly_dealers_2012.dta"
tab TaxPeriod if _merge==3
tab TaxPeriod if _merge==1
br TaxCreditBeforeAdjustment SumTaxAmount if _merge==3&TaxPeriod=="Apr-2012"
merge m:1 DealerTIN using "H:\Bogus_CancellationData.dta", keepusing(Reasons CancellationDate) generate(_merge_bogus2)
br TaxCreditBeforeAdjustment SumTaxAmount _merge_bogus2 if _merge==3&TaxPeriod=="Apr-2012"
br TaxCreditBeforeAdjustment SumTaxAmount _merge_bogus2 if _merge==3&TaxPeriod=="Apr-2012"&_merge_bogus2==3
gen ratio=TaxCreditBeforeAdjustment/SumTaxAmount
sum ratio
sum ratio if ratio!=.
br if ratio>1
br TaxCreditBeforeAdjustment SumTaxAmount _merge_bogus2 if _merge==3&TaxPeriod=="Apr-2012"&_merge_bogus2==3&ratio>3
br TaxCreditBeforeAdjustment SumTaxAmount _merge_bogus2 if _merge==3&TaxPeriod=="Apr-2012"&_merge_bogus2==3&ratio<2
br TaxCreditBeforeAdjustment SumTaxAmount _merge_bogus2 if _merge==3&TaxPeriod=="Apr-2012"&_merge_bogus2==3&ratio<.9
br TaxCreditBeforeAdjustment SumTaxAmount _merge_bogus2 if _merge==3&TaxPeriod=="Apr-2012"&tatio<.9
br TaxCreditBeforeAdjustment SumTaxAmount _merge_bogus2 if _merge==3&TaxPeriod=="Apr-2012"&ratio<.9
sum ration if _merge==3&TaxPeriod=="Apr-2012"
sum ratio if _merge==3&TaxPeriod=="Apr-2012"
sum ratio if _merge==3&TaxPeriod=="Apr-2012", detail
br TaxCreditBeforeAdjustment SumTaxAmount _merge_bogus2 if _merge==3&TaxPeriod=="Apr-2012"&ratio>2
br ratio TaxCreditBeforeAdjustment SumTaxAmount _merge_bogus2 if _merge==3&TaxPeriod=="Apr-2012"&ratio>2
sum ratio if _merge==3&TaxPeriod=="Apr-2012"&ratio!=., detail
tab _merge
sum TurnoverGross AggregateAmountPaid AmountDepositedByDealer if _merge==1
sum TurnoverGross AggregateAmountPaid AmountDepositedByDealer if _merge==3
sum TurnoverGross AggregateAmountPaid AmountDepositedByDealer if _merge==3&TaxPeriod=="Apr-2012"
sum TurnoverGross AggregateAmountPaid AmountDepositedByDealer if _merge==1&TaxPeriod=="Apr-2012"
sum TurnoverGross AggregateAmountPaid AmountDepositedByDealer if _merge==3&TaxPeriod=="FirstQuarter-2012"	
