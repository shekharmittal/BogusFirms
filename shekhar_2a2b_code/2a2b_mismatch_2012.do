use "E:\data\annexure_2A2B_monthly_201213.dta", clear

keep if SaleOrPurchase=="BF"
keep if SalePurchaseType=="LS"|SalePurchaseType==""

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
replace DealerGoodType=trim(DealerGoodType)

keep if DealerGoodType=="RD"|DealerGoodType=="OT"|DealerGoodType=="CG"

keep if TransactionType=="GD"|TransactionType=="WC"
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
replace Rate="20" if Rate=="20.00"
replace Rate="20" if Rate=="20.0"
replace Rate="5" if Rate=="0.05"
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

drop if Rate=="3.00"
drop if Rate=="2.50"
drop if SellerBuyerTIN=="NA"|SellerBuyerTIN=="0"
drop T985DF5 T985DF6 T985DF7 T985DF8 FormsStatus T985DF4
drop if Rate==""


bys SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen Count=_n
by SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount: gen TotalCount=_N

keep if Count==1
 drop Count TotalCount

isid SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount

save "F:\2a2b_analysis\MatchAnalysis\sales_2a2b_2012.dta", replace

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

save "F:\2a2b_analysis\MatchAnalysis\purchases_2a2b_2012.dta", replace

rename DealerTIN x
rename SellerBuyerTIN DealerTIN
rename x SellerBuyerTIN

merge 1:1 SourceFile TransactionType DealerTIN SellerBuyerTIN Rate Amount TaxAmount TotalAmount using "F:\2a2b_analysis\MatchAnalysis\sales_2a2b_2012.dta"

drop T985DF5 T985DF6 T985DF7 T985DF8 FormsStatus T985DF4


gen TaxMonth=25 if Month==4
replace TaxMonth=26 if Month==5
replace TaxMonth=27 if Month==6
replace TaxMonth=28 if Month==7
replace TaxMonth=29 if Month==8
replace TaxMonth=30 if Month==9
replace TaxMonth=31 if Month==10
replace TaxMonth=32 if Month==11
replace TaxMonth=33 if Month==12
replace TaxMonth=34 if Month==1
replace TaxMonth=35 if Month==2
replace TaxMonth=36 if Month==3

drop TotalCount


gen Match=0
replace Match=1 if _merge==3


gen TaxQuarter=0
replace TaxQuarter=9 if TaxMonth>24&TaxMonth<=27
replace TaxQuarter=10 if TaxMonth>27&TaxMonth<=30
replace TaxQuarter=11 if TaxMonth>30&TaxMonth<=33
replace TaxQuarter=12 if TaxMonth>33&TaxMonth<=36


//Purchases
drop if _merge==2


bys TaxMonth SellerBuyerTIN: egen TotalMatch=sum(Match)
by TaxMonth SellerBuyerTIN: egen AvgMatch=mean(Match)
by TaxMonth SellerBuyerTIN: gen RunningCount=_n
by TaxMonth SellerBuyerTIN: gen TotalCount=_N
bys TaxMonth: sum AvgMatch if RunningCount==1, detail

keep if RunningCount==1
drop RunningCount 

bys TaxQuarter SellerBuyerTIN: egen QAvgMatch=mean(AvgMatch)
by TaxQuarter SellerBuyerTIN: gen RunningCount=_n

keep if RunningCount==1
rename DealerTIN x
rename SellerBuyerTIN DealerTIN
rename x SellerBuyerTIN
replace AvgMatch=QAvgMatch
drop QAvgMatch
save "F:\2a2b_analysis\MatchAnalysis\purchases_2a2b_matchdata_2012.dta", replace


//Sales
drop if _merge==1


bys TaxMonth DealerTIN: egen TotalMatch=sum(Match)
by TaxMonth DealerTIN: egen AvgMatch=mean(Match)
by TaxMonth DealerTIN: gen RunningCount=_n
by TaxMonth DealerTIN: gen TotalCount=_N


bys TaxMonth: sum AvgMatch if RunningCount==1, detail

keep if RunningCount==1
drop RunningCount 

bys TaxQuarter DealerTIN: egen QAvgMatch=mean(AvgMatch)
by TaxQuarter DealerTIN: gen RunningCount=_n

keep if RunningCount==1
replace AvgMatch=QAvgMatch
drop QAvgMatch
save "F:\2a2b_analysis\MatchAnalysis\sales_2a2b_matchdata_2012.dta", replace


use "F:\2a2b_analysis\MatchAnalysis\sales_2a2b_matchdata_2012.dta", clear
append using "F:\2a2b_analysis\MatchAnalysis\sales_2a2b_matchdata_201314.dta"

drop Amount TaxAmount TotalAmount SaleOrPurchase SalePurchaseType DealerGoodType TransactionType Rate SellerBuyerTIN DealerName Month Year DateTime ReceiptId Date TaxRate AEBoolean T985DF1 T985DF2 T985DF3 TaxMonth Match RunningCount Count2
save "F:\2a2b_analysis\MatchAnalysis\MatchSaleAllQuarters.dta", replace


use "F:\2a2b_analysis\MatchAnalysis\purchases_2a2b_matchdata_2012.dta", clear
append using "F:\2a2b_analysis\MatchAnalysis\purchases_2a2b_matchdata_201314.dta"
keep if RunningCount==1

drop Amount TaxAmount TotalAmount SaleOrPurchase SalePurchaseType DealerGoodType TransactionType Rate SellerBuyerTIN DealerName Month Year DateTime ReceiptId Date TaxRate AEBoolean T985DF1 T985DF2 T985DF3 TaxMonth Match RunningCount Count2

save "F:\2a2b_analysis\MatchAnalysis\MatchPurchaseAllQuarters.dta", replace


preserve
collapse AvgMatch (median) MidAvgMatch=AvgMatch, by(TaxQuarter)
twoway (connected AvgMatch TaxQuarter)
restore


preserve
collapse AvgMatch, by(TaxQuarter BogusDummy)
twoway (connected AvgMatch TaxQuarter if BogusDumm==0) (connected AvgMatch TaxQuarter if BogusDumm==1)
restore


collapse AvgMatch , by(TaxQuarter)


twoway (connected AvgMatch TaxQuarter) (connected MidAvgMatch TaxQuarter), legend(on order(1 "Mean of Matches by Firm" 2 "Median of Matches by Firm")) title("Trends of Buyer and Seller Matches") note("Proportion calculated wrt purchases")
