cd "E:\data"

use "annexure_2A2B_quarterly_2013.dta", clear
use "annexure_2A2B_quarterly_2014.dta", clear

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

replace TransactionType=trim(TransactionType)
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
keep if DealerGoodType=="RD"

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
replace Rate="5" if Rate=="4.99"
replace Rate="4" if Rate=="4.00"

drop T985DF5 T985DF6 T985DF7 T985DF8 FormsStatus T985DF4

//bys SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen Count=_n
//by SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen TotalCount=_N

bys SourceFile DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen Count2=_n
by SourceFile DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen TotalCount2=_N

keep if Count2==1
drop if SellerBuyerTIN==""

isid SourceFile DealerTIN SellerBuyerTIN Rate Amount TaxAmount

append using "F:\2a2b_analysis\MatchAnalysis\sales_2a2b_2013.dta"

save "F:\2a2b_analysis\MatchAnalysis\sales_2a2b_2013.dta", replace

save "F:\2a2b_analysis\MatchAnalysis\sales_2a2b.dta", replace


/**************************
//Purchased From
***************************/
use "annexure_2A2B_quarterly_2013.dta", clear
use "annexure_2A2B_quarterly_2014.dta", clear

keep if SaleOrPurchase=="AE"

keep if SalePurchaseType=="CG"|SalePurchaseType=="OT"|DealerGoodType=="CG"|DealerGoodType=="OT"
drop T985DF5 T985DF6 T985DF7 T985DF8 FormsStatus T985DF4

replace TransactionType=trim(TransactionType)
replace TransactionType="GD" if TransactionType=="GD "
replace TransactionType="GD" if TransactionType=="GD  "
replace TransactionType="GD" if TransactionType=="GD    "
replace TransactionType="GD" if TransactionType=="GD     "
replace TransactionType="GD" if TransactionType=="GD             "
replace TransactionType="GD" if TransactionType=="Gd"
replace TransactionType="GD" if TransactionType=="gd"
replace TransactionType="GD" if TransactionType=="gd "
replace TransactionType="GD" if TransactionType=="gD"

replace TransactionType="WC" if TransactionType=="wc"
replace TransactionType="WC" if TransactionType=="Wc"

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

*gen dummy=1
*bys SourceFile DealerTIN SellerBuyerTIN:egen TransActionCount=sum(dummy)
*by SourceFile DealerTIN SellerBuyerTIN:egen SumAmount=sum(Amount)
*by SourceFile DealerTIN SellerBuyerTIN:egen SumTaxAmount=sum(TaxAmount)
*by SourceFile DealerTIN SellerBuyerTIN:egen SumTotalAmount=sum(TotalAmount)

drop if Rate=="13.13"|Rate=="14"
drop if SellerBuyerTIN==""

//bys SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen Count=_n
bys SourceFile DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen Count2=_n

//bys SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen TotalCount=_N
//bys SourceFile DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen TotalCount2=_N

keep if Count2==1

isid SourceFile DealerTIN SellerBuyerTIN Rate Amount TaxAmount

save "F:\2a2b_analysis\MatchAnalysis\purchases_2a2b_2013.dta", replace

append using "F:\2a2b_analysis\MatchAnalysis\purchases_2a2b_2013.dta"

save "F:\2a2b_analysis\MatchAnalysis\purchases_2a2b.dta", replace
	
rename DealerTIN x
rename SellerBuyerTIN DealerTIN
rename x SellerBuyerTIN

merge 1:1 SourceFile DealerTIN SellerBuyerTIN Rate Amount TaxAmount TotalAmount using "F:\2a2b_analysis\MatchAnalysis\sales_2a2b.dta", keepusing(Quarter)

//merge 1:1 SourceFile DealerTIN SellerBuyerTIN Rate Amount TaxAmount TotalAmount using "Z:\sales_2a2b_2013.dta", keepusing(Quarter)

//Purchases
drop if _merge==2

//Sales
drop if _merge==1

/*gen Match=0

replace Match=1 if _merge==3
bys SellerBuyerTIN: egen TotalMatch=sum(Match)
by SellerBuyerTIN: egen AvgMatch=mean(Match)

by SellerBuyerTIN: gen RunningCount=_n
sum AvgMatch if RunningCount==1, detail

histogram AvgMatch if RunningCount==1
histogram AvgMatch if RunningCount==1, fraction
*/

gen TaxQuarter=13 if SourceFile=="t9854113"
replace TaxQuarter=14 if SourceFile=="t9854213"
replace TaxQuarter=15 if SourceFile=="t9854313"
replace TaxQuarter=16 if SourceFile=="t9854413"
replace TaxQuarter=17 if SourceFile=="t9854114"
replace TaxQuarter=18 if SourceFile=="t9854214"
replace TaxQuarter=19 if SourceFile=="t9854314"
replace TaxQuarter=20 if SourceFile=="t9854414"


gen Match=0
replace Match=1 if _merge==3

//Purchases
bys TaxQuarter SellerBuyerTIN: egen TotalMatch=sum(Match)
by TaxQuarter SellerBuyerTIN: egen AvgMatch=mean(Match)
by TaxQuarter SellerBuyerTIN: gen RunningCount=_n
by TaxQuarter SellerBuyerTIN: gen TotalCount=_N

//Sales
bys TaxQuarter DealerTIN: egen TotalMatch=sum(Match)
by TaxQuarter DealerTIN: egen AvgMatch=mean(Match)
by TaxQuarter DealerTIN: gen RunningCount=_n
by TaxQuarter DealerTIN: gen TotalCount=_N



//Purchases
rename DealerTIN x
rename SellerBuyerTIN DealerTIN
rename x SellerBuyerTIN
merge m:1 DealerTIN using "H:\Bogus_CancellationData.dta", keepusing(Reasons CancellationDate) generate(_merge_bogus2)
rename DealerTIN x
rename SellerBuyerTIN DealerTIN
rename x SellerBuyerTIN

//Sales
merge m:1 DealerTIN using "H:\Bogus_CancellationData.dta", keepusing(Reasons CancellationDate) generate(_merge_bogus2)


drop if _merge_bogus2==2
gen SecondBogusDummy=0
replace SecondBogusDummy=1 if _merge_bogus2==3

log using "H:\2a2b_analysis\2a2b_mismatch_analysis.log", replace
bys TaxQuarter SecondBogusDummy: sum AvgMatch TotalCount TotalRevision if RunningCount==1, detail
loneway AvgMatch SellerBuyerTIN if RunningCount==1
log close

log using "H:\2a2b_analysis\2a2b_mismatch_analysis_sales.log"
bys TaxQuarter SecondBogusDummy: sum AvgMatch if RunningCount==1, detail
loneway AvgMatch DealerTIN if RunningCount==1
log close


rename DealerTIN x
rename SellerBuyerTIN DealerTIN
rename x SellerBuyerTIN
save "Z:\purchases_2a2b_matchdata_all.dta"



keep if RunningCount==1
save "F:\2a2b_analysis\MatchAnalysis\sales_2a2b_matchdata_201314.dta"




cd "D:\data"
use "form16_data_v3_0901.dta", clear
keep if TaxPeriod=="Third Quarter-2010"|TaxPeriod=="Third Quarter-2011"|TaxPeriod=="Third Quarter-2012"|TaxPeriod=="Third Quarter-2014"|TaxPeriod=="Third Quarter-2013"|TaxPeriod=="First Quarter-2010"|TaxPeriod=="First Quarter-2011"|TaxPeriod=="First Quarter-2012"|TaxPeriod=="First Quarter-2014"|TaxPeriod=="First Quarter-2013"|TaxPeriod=="Second Quarter-2010"|TaxPeriod=="Second Quarter-2011"|TaxPeriod=="Second Quarter-2012"|TaxPeriod=="Second Quarter-2014"|TaxPeriod=="Second Quarter-2013"|TaxPeriod=="Fourth Quarter-2010"|TaxPeriod=="Fourth Quarter-2011"|TaxPeriod=="Fourth Quarter-2012"|TaxPeriod=="Fourth Quarter-2014"|TaxPeriod=="Fourth Quarter-2013"


drop Tin T312203 T312202 Frequency T312020 T312021 DealerName DealerAddress DealerTelephone T312028 T312029 T312043 T312044 T312051 T312052 T312057 T312082 T312083 AccountNumber AccountType MICR BankDetails T312092 T312093 T312137 T312138 T312139 T312140 T312141 T312142 T312152 T312153 T312154 T312155 T312156 T312157 T312158 T312159 T312160 T312161 T312162 T312163 T312164 T312165 T312166 T312167
drop T312104 T312105 T312106 T312107 T312108 T312109 T312110 T312111 T312112 T312113 T312114 T312115 T312116 T312117 T312118 T312119 T312120 T312121 T312122 T312123 T312124 T312125 T312126 T312127 T312128 T312129 T312130 T312131 T312132 T312133 T312134 Name Designation Place Date TDSString Receipt2A2B T312170 T312DF1 T312DF2 T312DF3 T312DF4 T312DF5 T312DF6 T312DF7 T312DF8 T312178 Signatory


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

label define year 1 "2010-11" 2 "2011-12" 3 "2012-13" 4 "2013-14" 5 "2014-15" 6 "2015-16"
label define quarter 1 "Q1, 2010-11" 2 "Q2, 2010-11" 3 "Q3, 2010-11" 4 "Q4, 2010-11" 5 "Q1, 2011-12" 6 "Q2, 2011-12" 7 "Q3, 2011-12" 8 "Q4, 2011-12" 9 "Q1, 2012-13" 10 "Q2, 2012-13" 11 "Q3, 2012-13" 12 "Q4, 2012-13" 13 "Q1, 2013-14" 14 "Q2, 2013-14" 15 "Q3, 2013-14" 16 "Q4, 2013-14" 17 "Q1, 2014-15" 18 "Q2, 2014-15" 19 "Q3, 2014-15" 20 "Q4, 2014-15"

label values TaxQuarter quarter
label values TaxYear year	

gen MoneyDeposited=max(AggregateAmountPaid, AmountDepositedByDealer)
gen PositiveContribution=0
replace PositiveContribution=1 if MoneyDeposited>0

merge m:1 DealerTIN using "H:\Bogus_CancellationData.dta", keepusing(Reasons CancellationDate) generate(_merge_bogus2)
drop if _merge_bogus2==2
gen SecondBogusDummy=0
replace SecondBogusDummy=1 if _merge_bogus2==3


gsort DealerTIN TaxPeriod -id
by DealerTIN TaxPeriod: gen RevisionCount=_n
by DealerTIN TaxPeriod: gen TotalRevisionCount=_N

replace TotalRevisionCount=TotalRevisionCount-1

keep if RevisionCount==1
save "Z:\revision_count.dta"


use "Z:\purchases_2a2b_matchdata_all.dta", clear
keep if RunningCount==1
merge 1:1 DealerTIN TaxQuarter using "Z:\revision_count.dta", keepusing(SecondBogusDummy TotalRevisionCount PositiveContribution MoneyDeposited TurnoverGross) generate(_merge_revision)

drop if _merge_revision==1	

log using "H:\2a2b_analysis\2a2b_mismatch_revision_analysis.log"
corr TotalRevisionCount AvgMatch

reg AvgMatch TotalRevisionCount, cluster(DealerTIN)
reg AvgMatch TotalRevisionCount i.TaxQuarter, cluster(DealerTIN)

destring DealerTIN, replace
xtset DealerTIN TaxQuarter
xtreg AvgMatch TotalRevisionCount i.TaxQuarter, fe cluster(DealerTIN)
xtreg AvgMatch TotalRevisionCount SecondBogusDummy i.TaxQuarter, re cluster(DealerTIN)

xtreg AvgMatch TotalRevisionCount i.TaxQuarter, be 
xtreg AvgMatch TotalRevisionCount SecondBogusDummy i.TaxQuarter, be 

xtreg PositiveContribution AvgMatch i.TaxQuarter SecondBogusDummy, be

reg AvgMatch TotalRevisionCount SecondBogusDummy, cluster(DealerTIN)
reg AvgMatch TotalRevisionCount SecondBogusDummy i.TaxQuarter, cluster(DealerTIN)

ci AvgMatch if _merge_revision==3, by(TaxQuarter SecondBogusDummy)
log close

//To check correlations between seller and dealer matching averages
keep if RunningCount==1
keep TaxQuarter SellerBuyerTIN TotalMatch AvgMatch TotalCount
rename TotalMatch SellerTotalMatch
rename AvgMatch SellerAvgMatch
rename TotalCount SellerTotalCount
save "Z:\purchases_2a2b_sellers.dta"

merge m:1 TaxQuarter SellerBuyerTIN using "Z:\purchases_2a2b_sellers.dta", keepusing(SellerTotalMatch SellerAvgMatch SellerTotalCount) generate(_merge_avg)
tab _merge_avg
corr AvgMatch SellerAvgMatch if TaxQuarter==13
bys TaxQuarter: corr AvgMatch SellerAvgMatch
