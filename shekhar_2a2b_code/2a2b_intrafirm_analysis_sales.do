cd "E:\data"
use "annexure_2A2B_quarterly_2014.dta", clear

keep if SaleOrPurchase=="BF"
keep if SalePurchaseType=="LS"

keep if DealerGoodType=="RD"|DealerGoodType=="UD"


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
keep if Count==1
drop Count

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

save "F:\2a2b_analysis\IntraFirmAnalysis\quarterly_dealers_2014_sales.dta", replace



cd "D:\data"
use "annexure_2A2B_quarterly_2013.dta", clear

keep if SaleOrPurchase=="BF"
keep if SalePurchaseType=="LS"

keep if DealerGoodType=="RD"|DealerGoodType=="UD"


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


save "F:\2a2b_analysis\IntraFirmAnalysis\quarterly_dealers_2013_sales.dta", replace


use "E:\data\annexure_2A2B_monthly_201213.dta", clear
keep if SaleOrPurchase=="BF"
keep if SalePurchaseType=="LS"

replace DealerGoodType="OT" if DealerGoodType==" OT"
replace DealerGoodType="OT" if DealerGoodType=="OT             "
replace DealerGoodType="OT" if DealerGoodType=="OT     "
replace DealerGoodType="OT" if DealerGoodType=="OT "
replace DealerGoodType="OT" if DealerGoodType=="OT  "
replace DealerGoodType="OT" if DealerGoodType=="Ot"
replace DealerGoodType="OT" if DealerGoodType=="ot"
replace DealerGoodType="OT" if DealerGoodType=="ot "
replace DealerGoodType="OT" if DealerGoodType=="OT  "

replace DealerGoodType="UD" if DealerGoodType=="uD"
replace DealerGoodType="UD" if DealerGoodType=="ud"

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

drop if !(DealerGoodType=="RD"|DealerGoodType=="OT"|DealerGoodType=="CG"|DealerGoodType==""|DealerGoodType=="UD")
keep if TransactionType=="GD"|TransactionType=="WC"|TransactionType==""

drop if Rate==""

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
replace Rate="12.5" if Rate=="13.50"


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
drop Amount TaxAmount TotalAmount

save "F:\2a2b_analysis\IntraFirmAnalysis\monthly_dealers_2012_sales.dta", replace
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
save "F:\2a2b_analysis\IntraFirmAnalysis\monthly_2a2b_combined_quarterly_sales.dta"

append using "F:\2a2b_analysis\IntraFirmAnalysis\quarterly_dealers_2013_sales.dta"
append using "F:\2a2b_analysis\IntraFirmAnalysis\quarterly_dealers_2014_sales.dta"

drop SalePurchaseType DealerGoodType TransactionType DealerName SellerBuyerTIN Month Year DateTime Date FormsStatus TaxRate AEBoolean T985DF1 T985DF2 T985DF3 T985DF4 T985DF5 T985DF6 T985DF7 T985DF8 SourceFile 

save "F:\2a2b_analysis\IntraFirmAnalysis\Combined2a2b_allyears_sales.dta", replace



use "E:\data\HighTurnoverDealers2010.dta", clear
append using "E:\data\MediumTurnoverDealers2010.dta"

merge 1:1 DealerTIN TaxQuarter using "F:\2a2b_analysis\IntraFirmAnalysis\Combined2a2b_allyears_sales.dta", keepusing(TransActionCount SumAmount SumTaxAmount SumTotalAmount)
keep if _merge==3

gen SalesRatio=OutputTaxBeforeAdjustment/SumTaxAmount
drop if SalesRatio>100&SalesRatio!=.

replace OutputTaxBeforeAdjustment=OutputTaxBeforeAdjustment/1000000
replace SumTaxAmount=SumTaxAmount/1000000



preserve
collapse (mean) SalesRatio (sum) OutputTaxBeforeAdjustment SumTaxAmount, by(TaxQuarter)
twoway (connected SalesRatio TaxQuarter if SalesRatio!=.)
restore 


preserve
collapse (mean) SalesRatio OutputTaxBeforeAdjustment SumTaxAmount, by(TaxQuarter)
twoway (connected OutputTaxBeforeAdjustment TaxQuarter if SalesRatio!=.) (connected SumTaxAmount TaxQuarter if SalesRatio!=.) 
restore 




/* Now we do the analysis for all the firms*/
use "E:\data\PreliminaryAnalysis\returns\form16_data_v4_01122017.dta", clear

drop ReturnCount Time Tin T312203 T312202 Frequency T312020 T312021 DealerName DealerAddress DealerTelephone T312028 T312029 T312043 T312044 T312051 T312052 T312057 T312082 T312083 AccountNumber AccountType MICR BankDetails T312092 T312093 T312137 T312138 T312139 T312140 T312141 T312142 T312152 T312153 T312154 T312155 T312156 T312157 T312158 T312159 T312160 T312161 T312162 T312163 T312164 T312165 T312166 T312167
drop T312104 T312105 T312106 T312107 T312108 T312109 T312110 T312111 T312112 T312113 T312114 T312115 T312116 T312117 T312118 T312119 T312120 T312121 T312122 T312123 T312124 T312125 T312126 T312127 T312128 T312129 T312130 T312131 T312132 T312133 T312134 Name Designation Place Date TDSString Receipt2A2B T312170 T312DF1 T312DF2 T312DF3 T312DF4 T312DF5 T312DF6 T312DF7 T312DF8 Days T312178 Signatory

drop if TaxPeriod=="Annual-2012"|TaxPeriod=="First Halfyear-2012"|TaxPeriod=="Second Halfyear-2012"|TaxPeriod=="Apr-2013"|TaxPeriod=="May-2013"

gen TaxYear=0
replace TaxYear=1 if TaxPeriod=="Annual-2010"
replace TaxYear=2 if TaxPeriod=="Annual-2011"
replace TaxYear=3 if TaxPeriod=="Annual-2012"

gen TaxHalfyear=0
replace TaxHalfyear=1 if TaxPeriod=="First Halfyear-2010"
replace TaxHalfyear=2 if TaxPeriod=="Second Halfyear-2010"
replace TaxHalfyear=3 if TaxPeriod=="First Halfyear-2011"
replace TaxHalfyear=4 if TaxPeriod=="Second Halfyear-2011"
replace TaxHalfyear=5 if TaxPeriod=="First Halfyear-2012"
replace TaxHalfyear=6 if TaxPeriod=="Second Halfyear-2012"

replace TaxYear=1 if TaxPeriod=="First Halfyear-2010"|TaxPeriod=="Second Halfyear-2010"
replace TaxYear=2 if TaxPeriod=="First Halfyear-2011"|TaxPeriod=="Second Halfyear-2011"
replace TaxYear=3 if TaxPeriod=="First Halfyear-2012"|TaxPeriod=="Second Halfyear-2012"


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

gsort DealerTIN TaxYear TaxHalfyear TaxQuarter TaxMonth

gen MoneyDeposited=max(AggregateAmountPaid, AmountDepositedByDealer)
gen TotalPurchases=PurchaseCapitalGoods+PurchaseOtherGoods+PurchaseUnregisteredDealer
gen PercValueAdded=(TurnoverGross-TotalPurchases)/(TotalPurchases)
gen TotalValueAdded=(TurnoverGross-TotalPurchases)
gen PercPurchaseUnregisteredDealer=PurchaseUnregisteredDealer/(TotalPurchases)
*label variable PurchaseNoCredit "R6.3 Total local purchases that are no eligible for credit of input tax"
label variable PercValueAdded "A measure of fraction of value added, definition is (TurnoverGross-PurchaseCapitalGoods-PurchaseOtherGoods-PurchaseUnregisteredDealer)/(PurchaseCapitalGoods+PurchaseOtherGoods+PurchaseNoCredit)"
label variable PercPurchaseUnregisteredDealer "A measure of amount purchased from unregistered dealers:PurchaseUnregisteredDealer/(PurchaseCapitalGoods+PurchaseOtherGoods+PurchaseUnregisterdDealer)"
label variable TotalValueAdded "Total value added, (TurnoverGross-TotalPurchases)"
label variable TotalPurchases "Total purchases made: PurchaseCapitalGoods+PurchaseOtherGoods+PurchaseUnregisteredDealer"

gen PositiveContribution=0
replace PositiveContribution=1 if MoneyDeposited>0


gsort DealerTIN TaxYear TaxHalfyear TaxQuarter TaxMonth
gen AnnualDummy=1 if TaxPeriod=="Annual-2010"|TaxPeriod=="Annual-2011"
gen SemiAnnualDummy=1 if TaxPeriod=="First Halfyear-2010"|TaxPeriod=="First Halfyear-2011"|TaxPeriod=="Second Halfyear-2010"|TaxPeriod=="Second Halfyear-2011"
gen QuarterlyDummy=1 if TaxPeriod=="First Quarter-2010"|TaxPeriod=="First Quarter-2011"|TaxPeriod=="Second Quarter-2010"|TaxPeriod=="Second Quarter-2011"|TaxPeriod=="Third Quarter-2010"|TaxPeriod=="Third Quarter-2011"|TaxPeriod=="Fourth Quarter-2010"|TaxPeriod=="Fourth Quarter-2011"
gen MonthlyDummy=1 if TaxPeriod=="Apr-2010"|TaxPeriod=="Apr-2011"|TaxPeriod=="Apr-2012"|TaxPeriod=="Aug-2010"|TaxPeriod=="Aug-2011"|TaxPeriod=="Aug-2012"|TaxPeriod=="Dec-2010"|TaxPeriod=="Dec-2011"|TaxPeriod=="Dec-2012"|TaxPeriod=="Feb-2011"|TaxPeriod=="Feb-2012"|TaxPeriod=="Feb-2013"|TaxPeriod=="Jan-2011"|TaxPeriod=="Jan-2012"|TaxPeriod=="Jan-2013"|TaxPeriod=="Jul-2010"|TaxPeriod=="Jul-2011"|TaxPeriod=="Jul-2012"|TaxPeriod=="Jun-2010"|TaxPeriod=="Jun-2011"|TaxPeriod=="Jun-2012"|TaxPeriod=="Mar-2011"|TaxPeriod=="Mar-2012"|TaxPeriod=="Mar-2013"|TaxPeriod=="May-2010"|TaxPeriod=="May-2011"|TaxPeriod=="May-2012"|TaxPeriod=="Nov-2010"|TaxPeriod=="Nov-2011"|TaxPeriod=="Nov-2012"|TaxPeriod=="Oct-2010"|TaxPeriod=="Oct-2011"|TaxPeriod=="Oct-2012"|TaxPeriod=="Sep-2010"|TaxPeriod=="Sep-2011"|TaxPeriod=="Sep-2012"


by DealerTIN: replace AnnualDummy=AnnualDummy[_n-1] if AnnualDummy>=.
by DealerTIN: replace SemiAnnualDummy=SemiAnnualDummy[_n-1] if SemiAnnualDummy>=.
by DealerTIN: replace QuarterlyDummy=QuarterlyDummy[_n-1] if QuarterlyDummy>=.
by DealerTIN: replace MonthlyDummy=MonthlyDummy[_n-1] if MonthlyDummy>=.


drop if AnnualDummy==1&SemiAnnualDummy==1&TaxYear==1
drop if AnnualDummy==1&QuarterlyDummy==1&TaxYear==1
drop if AnnualDummy==1&MonthlyDummy==1&TaxYear==1
drop if SemiAnnualDummy==1&QuarterlyDummy==1&TaxYear==1
drop if SemiAnnualDummy==1&MonthlyDummy==1&TaxYear==1
drop if QuarterlyDummy==1&MonthlyDummy==1&TaxYear==1
drop AnnualDummy SemiAnnualDummy MonthlyDummy QuarterlyDummy


gsort DealerTIN TaxYear TaxHalfyear TaxQuarter TaxMonth
gen AnnualDummy=1 if TaxPeriod=="Annual-2011"
gen SemiAnnualDummy=1 if TaxPeriod=="First Halfyear-2011"|TaxPeriod=="Second Halfyear-2011"
gen QuarterlyDummy=1 if TaxPeriod=="First Quarter-2011"|TaxPeriod=="Second Quarter-2011"|TaxPeriod=="Third Quarter-2011"|TaxPeriod=="Fourth Quarter-2011"
gen MonthlyDummy=1 if TaxPeriod=="Apr-2011"|TaxPeriod=="Aug-2011"|TaxPeriod=="Dec-2011"|TaxPeriod=="Feb-2012"|TaxPeriod=="Jan-2012"|TaxPeriod=="Jul-2011"|TaxPeriod=="Jun-2011"|TaxPeriod=="Mar-2012"|TaxPeriod=="May-2011"|TaxPeriod=="Nov-2011"|TaxPeriod=="Oct-2011"|TaxPeriod=="Sep-2011"

by DealerTIN: replace AnnualDummy=AnnualDummy[_n-1] if AnnualDummy>=.
by DealerTIN: replace SemiAnnualDummy=SemiAnnualDummy[_n-1] if SemiAnnualDummy>=.
by DealerTIN: replace QuarterlyDummy=QuarterlyDummy[_n-1] if QuarterlyDummy>=.
by DealerTIN: replace MonthlyDummy=MonthlyDummy[_n-1] if MonthlyDummy>=.

drop if AnnualDummy==1&SemiAnnualDummy==1&TaxYear==2
drop if AnnualDummy==1&QuarterlyDummy==1&TaxYear==2
drop if AnnualDummy==1&MonthlyDummy==1&TaxYear==2
drop if SemiAnnualDummy==1&QuarterlyDummy==1&TaxYear==2
drop if SemiAnnualDummy==1&MonthlyDummy==1&TaxYear==2
drop if QuarterlyDummy==1&MonthlyDummy==1&TaxYear==2
drop AnnualDummy SemiAnnualDummy MonthlyDummy QuarterlyDummy


gsort DealerTIN TaxYear TaxHalfyear TaxQuarter TaxMonth
gen QuarterlyDummy=1 if TaxPeriod=="First Quarter-2012"|TaxPeriod=="Second Quarter-2012"|TaxPeriod=="Third Quarter-2012"|TaxPeriod=="Fourth Quarter-2012"
gen MonthlyDummy=1 if TaxPeriod=="Apr-2012"|TaxPeriod=="Aug-2012"|TaxPeriod=="Dec-2012"|TaxPeriod=="Feb-2013"|TaxPeriod=="Jan-2013"|TaxPeriod=="Jul-2012"|TaxPeriod=="Jun-2012"|TaxPeriod=="Mar-2013"|TaxPeriod=="May-2012"|TaxPeriod=="Nov-2012"|TaxPeriod=="Oct-2012"|TaxPeriod=="Sep-2012"


by DealerTIN: replace QuarterlyDummy=QuarterlyDummy[_n-1] if QuarterlyDummy>=.
by DealerTIN: replace MonthlyDummy=MonthlyDummy[_n-1] if MonthlyDummy>=.
drop if QuarterlyDummy==1&MonthlyDummy==1&TaxYear==3


//drop if TaxPeriod=="Second Halfyear-2010"|TaxPeriod=="May-2013"|TaxPeriod=="First Halfyear-2010"|TaxPeriod=="First Halfyear-2011"|TaxPeriod=="Second Halfyear-2011"|TaxPeriod=="First Halfyear-2012"|TaxPeriod=="Second Halfyear-2012"|TaxPeriod=="Annual-2010"|TaxPeriod=="Annual-2011"|TaxPeriod=="Annual-2012"|TaxPeriod=="Apr-2013"|TaxPeriod=="Apr-2010"|TaxPeriod=="May-2010"|TaxPeriod=="Jun-2010"

drop if TurnoverGross<0

drop if TaxPeriod=="Annual-2010"|TaxPeriod=="Annual-2011"|TaxPeriod=="First Halfyear-2011"|TaxPeriod=="Second Halfyear-2011"
drop if TaxYear<3
drop  DieselPurchase PurchaseCompositionDealer PurchaseNonCreditableGoods PurchaseRetailInvoice PurchaseTaxFreeGoods WCLabourPurchase PurchaseIneligibleForITC PurchaseDelhiFormH PurchaseCapitalNonCreditGood
drop T312002 RefundClaimedBoolean PeriodBegin PeriodEnd TurnoverAt1 OutputTaxAt1 TurnoverAt5 OutputTaxAt5 TurnoverAt125 OutputTaxAt125 TurnoverAt20 OutputTaxAt20 WCTurnoverAt5 WCOutputTaxAt5 WCTurnoverAt125 WCOutputTaxAt125 ExemptedSales PurchaseGoodsAt1 CreditGoodsAt1 PurchaseGoodsAt5 CreditGoodsAt5 PurchaseGoodsAt125 CreditGoodsAt125 PurchaseGoodsAt20 CreditGoodsAt20 WCPurchaseAt5 WCCreditAt5 WCPurchaseAt125 WCCreditAt125 InterestPayable PenaltyPayable TDSCertificates
drop HighSeaPurchase HighSeaSale PurchaseExemptedGoodsCST SaleExemptedGoodsCST PurchaseExemptedGoodsSchedule SaleExemptedGoodsSchedule OtherDealersGoodsReJobF OtherDealersGoodsTrJobF OwnGoodsReceivedAfterJobF OwnGoodsTransferredAfterJobF InwardStockTransferConsignmentF OutwardStockTransferConsignmentF InwardStockTransferBranchF OutwardStockTransferBranchF CSTAdjustmentVAT CarryForwardTaxCredit SaleDelhiFormH LandCharges LaborCharges DieselSale ReturnType 
drop SourceFile AcknowledgementNumber ApprovalDate Status TDSDetails WardName T312011 T312012 T312013 T312149 T312150 T312151

collapse (mean)TaxYear (sum) PurchaseUnregisteredDealer TurnoverGross TurnoverCentral TurnoverLocal PurchaseOtherGoods OutputTaxBeforeAdjustment AdjustmentOutputTax TotalOutputTax PurchaseCapitalGoods CreditCapitalGoods CreditOtherGoods TaxCreditBeforeAdjustment AdjustmentTaxCredit TotalTaxCredit NetTax BalancePayable AmountDepositedByDealer AggregateAmountPaid NetBalance BalanceBroughtForward AdjustCSTLiability RefundClaimed BalanceCarriedNextTaxPeriod InterStateSaleCD InterStatePurchaseCD InterStateSaleCE1E2 InterStatePurchaseCE1E2 InterStateExportsAgainstH InterStateImportsAgainstH InterStateExportsAgainstI InterStateImportsAgainstI InterStateExportsAgainstJ InterStateImportsAgainstJ ExportFromIndia ImportToIndia InterStateSaleOther InterStatePurchaseOther InterStateSaleCapital InterStatePurchaseCapital TotalInterStateSale TotalInterStatePurchase, by(DealerTIN TaxQuarter)

merge 1:1 DealerTIN TaxQuarter using "F:\2a2b_analysis\IntraFirmAnalysis\Combined2a2b_allyears_sales.dta", keepusing(TransActionCount SumAmount SumTaxAmount SumTotalAmount)

gen SalesRatio=OutputTaxBeforeAdjustment/SumTaxAmount
replace OutputTaxBeforeAdjustment=OutputTaxBeforeAdjustment/1000000
replace SumTaxAmount=SumTaxAmount/1000000

drop if SalesRatio>100&SalesRatio!=.
drop if _merge!=3


preserve
collapse (mean) SalesRatio (sum) OutputTaxBeforeAdjustment SumTaxAmount, by(TaxQuarter)
twoway (connected SalesRatio TaxQuarter if SalesRatio!=.)
graph save Graph "F:\2a2b_analysis\IntraFirmAnalysis\MeanSalesRatio_allfirms.gph"
//graph export "F:\2a2b_analysis\IntraFirmAnalysis\MeanPurchaseRatio_allfirms.pdf", as(pdf) replace
restore 


preserve
collapse (mean) SalesRatio (mean) OutputTaxBeforeAdjustment SumTaxAmount, by(TaxQuarter)
twoway (connected OutputTaxBeforeAdjustment TaxQuarter if SalesRatio!=.) (connected SumTaxAmount TaxQuarter if SalesRatio!=.) 
graph save Graph "F:\2a2b_analysis\IntraFirmAnalysis\OutputTaxComparison_allfirms.gph", replace
//graph export "F:\2a2b_analysis\IntraFirmAnalysis\TaxCreditComparison_allfirms.pdf", as(pdf) replace
restore 






/*Rough*/
{

use "Z:\monthly_dealers_2012_sales.dta", clear
merge m:m DealerTIN TaxQuarter Month using "Z:\DealerType.dta", keepusing(TypeDealer TaxQuarter TaxMonth)
keep if TypeDealer=="Monthly"&_merge==3
drop Amount TaxAmount TotalAmount Rate
save "Z:\monthly_2a2b_combined_sales.dta"

use "Z:\monthly_dealers_2012_sales.dta", clear
merge m:m DealerTIN TaxQuarter using "Z:\DealerType.dta", keepusing(TypeDealer TaxQuarter TaxMonth)
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
save "Z:\quarterly_2a2b_combined_sales.dta", replace


use "Z:\monthly_2a2b_combined_sales.dta", clear
append using "Z:\quarterly_2a2b_combined_sales.dta"
save "Z:\Combined2a2b_sales.dta", replace

use "Z:\monthly_2a2b_combined_sales.dta", clear
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
save "Z:\monthly_2a2b_combined_sales_quarterly.dta"
append using "Z:\quarterly_2a2b_combined_sales.dta"
save "Z:\Combined2a2b_sales_quarterly.dta"


use "D:\data\HighTurnoverDealers2010_with_distances.dta", clear
gen TypeDealer="Monthly"
append using "D:\data\MediumTurnoverDealers2010_with_distances.dta"
replace TypeDealer="Quarterly" if TypeDealer==""

bys DealerTIN TaxQuarter: gen Count=_N
drop if Count==2

merge m:1 DealerTIN using "H:\Bogus_CancellationData.dta", keepusing(Reasons CancellationDate) generate(_merge_bogus2)
drop if _merge_bogus2==2
gen SecondBogusDummy=0
replace SecondBogusDummy=1 if _merge_bogus2==3
drop _merge_bogus2

merge 1:1 DealerTIN  TypeDealer TaxQuarter using "Z:\Combined2a2b_sales_quarterly.dta", keepusing(TransActionCount SumAmount SumTaxAmount SumTotalAmount) generate(_merge_combine)
rename TransActionCount TransActionCount1 
rename SumAmount SumAmount1 
rename SumTaxAmount SumTaxAmount1
rename SumTotalAmount SumTotalAmount1
drop if _merge_combine==2



merge 1:1 DealerTIN TaxQuarter using "Z:\quarterly_dealers_2013_sales.dta", keepusing(TransActionCount SumAmount SumTaxAmount SumTotalAmount) generate(_merge_combine2)
replace TransActionCount=TransActionCount1 if TransActionCount==.
replace SumAmount=SumAmount1 if SumAmount==.
replace SumTaxAmount=SumTaxAmount1 if SumTaxAmount==.
replace SumTotalAmount=SumTotalAmount1 if SumTotalAmount==.

rename TransActionCount TransActionCount2 
rename SumAmount SumAmount2
rename SumTaxAmount SumTaxAmount2
rename SumTotalAmount SumTotalAmount2
drop if _merge_combine2==2


merge 1:1 DealerTIN TaxQuarter using "Z:\quarterly_dealers_2014_sales.dta", keepusing(TransActionCount SumAmount SumTaxAmount SumTotalAmount) generate(_merge_combine3)
drop if  _merge_combine3==2

replace TransActionCount=TransActionCount2 if TransActionCount==.
replace SumAmount=SumAmount2 if SumAmount==.
replace SumTaxAmount=SumTaxAmount2 if SumTaxAmount==.
replace SumTotalAmount=SumTotalAmount2 if SumTotalAmount==.

gen SalesRatio=OutputTaxBeforeAdjustment/SumTaxAmount

gen Dummy=.
replace Dummy=1 if SalesRatio!=.
gsort DealerTIN Dummy
by DealerTIN: replace Dummy=Dummy[_n-1] if Dummy>=.
replace Dummy=0 if Dummy==.


gen Dummy=0
replace Dummy=1 if SalesRatio!=.

drop if SalesRatio>100&SalesRatio!=.
replace OutputTaxBeforeAdjustment=OutputTaxBeforeAdjustment/1000000
replace SumTaxAmount=SumTaxAmount/1000000
replace MoneyDeposited=MoneyDeposited/1000000

collapse (mean) SalesRatio (sum) MoneyDeposited OutputTaxBeforeAdjustment SumTaxAmount, by(TaxQuarter Dummy)

twoway (connected MoneyDeposited TaxQuarter if Dummy==0) (connected OutputTaxBeforeAdjustment TaxQuarter if Dummy==1) (connected SumTaxAmount TaxQuarter if Dummy==1&TaxQuarter>8) (connected OutputTaxBeforeAdjustment TaxQuarter if Dummy==0) (connected SalesRatio TaxQuarter if Dummy==1, yaxis(2))


twoway (connected OutputTaxBeforeAdjustment TaxQuarter if Dummy==1) (connected SumTaxAmount TaxQuarter if Dummy==1&TaxQuarter>8) (connected OutputTaxBeforeAdjustment TaxQuarter if Dummy==0) (connected SalesRatio TaxQuarter if Dummy==1, yaxis(2)), legend(order (1 "Output Tax (consolidated returns)" 2 "Output Tax (in annexures)" 3 "Output Tax (without annexures)" 4 "Ratio of Consolidated and Annexure")) ytitle("Output Tax(in million)") xline(9)

drop _merge
merge m:1 DealerTIN using "D:\data\DataVerification\step3\DealerProfile_uniqueTin.dta", keepusing(Nature Constitution RegistrationType RegistrationDate SubmissionDate Ward BooleanInterState Boolean201011 Boolean201112 Boolean201213 BooleanThirdPartyStorage BooleanSurveyFilled GTONil201213 PhysicalWard BooleanRegisteredIEC BooleanRegisteredCE BooleanServiceTax)
keep if _merge==1|_merge==3


gen Government=.
replace Government=1 if Constitution=="Government Company"
replace Government=1 if Constitution=="Government Corporation"
replace Government=1 if Constitution=="Government Department"
replace Government=1 if Constitution=="Public Sector UnderTaking"


replace Government=0 if Constitution=="HUF"
replace Government=0 if Constitution=="Partnership"
replace Government=0 if Constitution=="Private Ltd. Company"
replace Government=0 if Constitution=="Proprietorship"

replace Government=2 if Constitution=="Public Ltd. Company"


collapse (mean) SalesRatio (sum) OutputTaxBeforeAdjustment MoneyDeposited SumTaxAmount, by(TaxQuarter Government Dummy )

twoway (connected SalesRatio TaxQuarter if Government==0) (connected SalesRatio TaxQuarter if Government==1) (connected SalesRatio TaxQuarter if Government==2) if TaxQuarter>8, legend(order (1 "Private firms" 2 "Government firms" 3 "Public limited firms"))  


twoway (connected OutputTaxBeforeAdjustment TaxQuarter if Dummy==1) (connected SumTaxAmount TaxQuarter if Dummy==1) (connected OutputTaxBeforeAdjustment TaxQuarter if Dummy==0) (connected SalesRatio TaxQuarter if Dummy==1, yaxis(2)) if Government==2, ytitle("Tax Credit (in million)") legend(order (1 "Taxcredit (consolidated returns)" 2 "Taxcredit (in annexures)" 3 "Taxcredit (without annexures)" 4 "Ratio of Consolidated and Annexure")) 


xtile MedianTurnover=TurnoverGross if TaxQuarter==1, nq(2)
xtile MedianMoney=MoneyDeposited if TaxQuarter==1, nq(2)
xtile QuantileCredit=TaxCreditBeforeAdjustment if TaxQuarter==1, nq(4)


gsort DealerTIN TaxQuarter
by DealerTIN: replace MedianTurnover=MedianTurnover[_n-1] if MedianTurnover>=.
by DealerTIN: replace MedianMoney=MedianMoney[_n-1] if MedianMoney>=.
by DealerTIN: replace QuantileCredit=QuantileCredit[_n-1] if QuantileCredit>=.


collapse (mean) SalesRatio (sum) OutputTaxBeforeAdjustment MoneyDeposited SumTaxAmount, by(TaxQuarter MedianTurnover Dummy)

collapse (mean) SalesRatio (sum) OutputTaxBeforeAdjustment MoneyDeposited SumTaxAmount, by(TaxQuarter MedianMoney Dummy)

collapse (mean) SalesRatio (sum) OutputTaxBeforeAdjustment MoneyDeposited SumTaxAmount, by(TaxQuarter QuantileCredit Dummy)


twoway (connected OutputTaxBeforeAdjustment TaxQuarter if Dummy==1) (connected SumTaxAmount TaxQuarter if Dummy==1&TaxQuarter>8) (connected OutputTaxBeforeAdjustment TaxQuarter if Dummy==0) (connected SalesRatio TaxQuarter if Dummy==1, yaxis(2)) if MedianTurnover==1, ytitle("Tax Credit (in million)") legend(order (1 "Output Tax (consolidated returns)" 2 "Output Tax (in annexures)" 3 "Output Tax (without annexures)" 4 "Ratio of Consolidated and Annexure")) ytitle("Output Tax(in million)") xline(9) title("Firms below median turnover")
graph save "H:\2a2b_analysis\IntrafirmAnalysis\MedianTurnover1_sales.gph"
twoway (connected OutputTaxBeforeAdjustment TaxQuarter if Dummy==1) (connected SumTaxAmount TaxQuarter if Dummy==1&TaxQuarter>8) (connected OutputTaxBeforeAdjustment TaxQuarter if Dummy==0) (connected SalesRatio TaxQuarter if Dummy==1, yaxis(2)) if MedianTurnover==2, ytitle("Tax Credit (in million)") legend(order (1 "Output Tax (consolidated returns)" 2 "Output Tax (in annexures)" 3 "Output Tax (without annexures)" 4 "Ratio of Consolidated and Annexure")) ytitle("Output Tax(in million)") xline(9) title("Firms above median turnover")
graph save "H:\2a2b_analysis\IntrafirmAnalysis\MedianTurnover2_sales.gph"
graph combine "H:\2a2b_analysis\IntrafirmAnalysis\MedianTurnover1_sales.gph" "H:\2a2b_analysis\IntrafirmAnalysis\MedianTurnover2_sales.gph"

legend(order (1 "Taxcredit (consolidated returns)" 2 "Taxcredit (in annexures)" 3 "Taxcredit (without annexures)" 4 "Ratio of Consolidated and Annexure")) 

}
