#include "sierrachart.h"
#include <cmath>

SCDLLName("AdvancedOrderFlowStrategy")

SCSFExport scsf_AdvancedOrderFlowStrategy(SCStudyInterfaceRef sc)
{
    SCSubgraphRef Subgraph_CumulativeDelta = sc.Subgraph[0];
    SCSubgraphRef Subgraph_VolumeImbalance = sc.Subgraph[1];
    SCSubgraphRef Subgraph_FVG = sc.Subgraph[2];
    SCSubgraphRef Subgraph_LongerTimeframeTrend = sc.Subgraph[3];
    SCSubgraphRef Subgraph_BuyEntry = sc.Subgraph[4];
    SCSubgraphRef Subgraph_SellEntry = sc.Subgraph[5];
    SCSubgraphRef Subgraph_ATR = sc.Subgraph[6];

    SCInputRef Input_Enabled = sc.Input[0];
    SCInputRef Input_CumulativeDeltaThreshold = sc.Input[1];
    SCInputRef Input_VolumeImbalanceThreshold = sc.Input[2];
    SCInputRef Input_FVGThreshold = sc.Input[3];
    SCInputRef Input_RiskPercentage = sc.Input[4];
    SCInputRef Input_MaxPositionSize = sc.Input[5];
    SCInputRef Input_TakeProfitMultiplier = sc.Input[6];
    SCInputRef Input_MaxHoldingTime = sc.Input[7];
    SCInputRef Input_VolatilityThreshold = sc.Input[8];
    SCInputRef Input_TradingStartTime = sc.Input[9];
    SCInputRef Input_TradingEndTime = sc.Input[10];
    SCInputRef Input_LongerTimeFrame = sc.Input[11];
    SCInputRef Input_SendOrdersToTradeService = sc.Input[12];

    if (sc.SetDefaults)
    {
        sc.GraphName = "Advanced Order Flow Strategy";
        sc.StudyDescription = "Professional-grade order flow trading strategy using cumulative delta, volume imbalances, and FVGs";
        
        sc.AutoLoop = 1;
        sc.GraphRegion = 0;
        sc.CalculationPrecedence = LOW_PREC_LEVEL;
        
        Subgraph_CumulativeDelta.Name = "Cumulative Delta";
        Subgraph_CumulativeDelta.DrawStyle = DRAWSTYLE_LINE;
        Subgraph_CumulativeDelta.PrimaryColor = RGB(0, 255, 0);
        
        Subgraph_VolumeImbalance.Name = "Volume Imbalance";
        Subgraph_VolumeImbalance.DrawStyle = DRAWSTYLE_BAR;
        Subgraph_VolumeImbalance.PrimaryColor = RGB(255, 0, 0);
        
        Subgraph_FVG.Name = "Fair Value Gap";
        Subgraph_FVG.DrawStyle = DRAWSTYLE_POINT;
        Subgraph_FVG.PrimaryColor = RGB(0, 0, 255);
        
        Subgraph_LongerTimeframeTrend.Name = "Longer Timeframe Trend";
        Subgraph_LongerTimeframeTrend.DrawStyle = DRAWSTYLE_LINE;
        Subgraph_LongerTimeframeTrend.PrimaryColor = RGB(255, 165, 0);
        
        Subgraph_BuyEntry.Name = "Buy Entry";
        Subgraph_BuyEntry.DrawStyle = DRAWSTYLE_ARROW_UP;
        Subgraph_BuyEntry.PrimaryColor = RGB(0, 255, 0);
        
        Subgraph_SellEntry.Name = "Sell Entry";
        Subgraph_SellEntry.DrawStyle = DRAWSTYLE_ARROW_DOWN;
        Subgraph_SellEntry.PrimaryColor = RGB(255, 0, 0);

        Subgraph_ATR.Name = "ATR";
        Subgraph_ATR.DrawStyle = DRAWSTYLE_IGNORE;
        
        Input_Enabled.Name = "Strategy Enabled";
        Input_Enabled.SetYesNo(0);
        
        Input_CumulativeDeltaThreshold.Name = "Cumulative Delta Threshold";
        Input_CumulativeDeltaThreshold.SetFloat(1000.0f);
        
        Input_VolumeImbalanceThreshold.Name = "Volume Imbalance Threshold";
        Input_VolumeImbalanceThreshold.SetFloat(0.7f);
        
        Input_FVGThreshold.Name = "FVG Threshold (in ticks)";
        Input_FVGThreshold.SetFloat(5.0f);
        
        Input_RiskPercentage.Name = "Risk Percentage per Trade";
        Input_RiskPercentage.SetFloat(1.0f);
        
        Input_MaxPositionSize.Name = "Maximum Position Size";
        Input_MaxPositionSize.SetInt(10);
        
        Input_TakeProfitMultiplier.Name = "Take Profit Multiplier";
        Input_TakeProfitMultiplier.SetFloat(2.0f);
        
        Input_MaxHoldingTime.Name = "Max Holding Time (in minutes)";
        Input_MaxHoldingTime.SetInt(120);
        
        Input_VolatilityThreshold.Name = "Volatility Threshold";
        Input_VolatilityThreshold.SetFloat(1.5f);
        
        Input_TradingStartTime.Name = "Trading Start Time";
        Input_TradingStartTime.SetTime(HMS_TIME(9, 30, 0));
        
        Input_TradingEndTime.Name = "Trading End Time";
        Input_TradingEndTime.SetTime(HMS_TIME(16, 0, 0));
        
        Input_LongerTimeFrame.Name = "Longer Timeframe (in minutes)";
        Input_LongerTimeFrame.SetInt(60);

        Input_SendOrdersToTradeService.Name = "Send Orders to Trade Service";
        Input_SendOrdersToTradeService.SetYesNo(0);
        
        sc.AllowMultipleEntriesInSameDirection = 0;
        sc.MaximumPositionAllowed = Input_MaxPositionSize.GetInt();
        sc.SupportReversals = 0;
        sc.SendOrdersToTradeService = 0;
        sc.AllowOppositeEntryWithOpposingPositionOrOrders = 0;
        sc.SupportAttachedOrdersForTrading = 1;
        sc.CancelAllOrdersOnEntriesAndReversals = 1;
        sc.AllowEntryWithWorkingOrders = 0;
        sc.CancelAllWorkingOrdersOnExit = 1;
        sc.AllowOnlyOneTradePerBar = 1;
        sc.MaintainTradeStatisticsAndTradesData = 1;
        
        return;
    }
    
    if (!Input_Enabled.GetYesNo())
        return;

    sc.SendOrdersToTradeService = Input_SendOrdersToTradeService.GetYesNo();
    
    sc.ATR(sc.BaseDataIn, Subgraph_ATR, sc.Index, 14, MOVAVGTYPE_SIMPLE);

    for (int i = sc.UpdateStartIndex; i < sc.ArraySize; i++)
    {
        float askVolume = sc.AskVolume[i];
        float bidVolume = sc.BidVolume[i];
        float delta = askVolume - bidVolume;
        
        Subgraph_CumulativeDelta[i] = (i == 0) ? delta : Subgraph_CumulativeDelta[i-1] + delta;
        
        float totalVolume = askVolume + bidVolume;
        float imbalanceRatio = 0;
        
        if (totalVolume > 0)
        {
            imbalanceRatio = fmaxf(askVolume, bidVolume) / totalVolume;
            Subgraph_VolumeImbalance[i] = (imbalanceRatio > Input_VolumeImbalanceThreshold.GetFloat()) 
                ? ((askVolume > bidVolume) ? imbalanceRatio : -imbalanceRatio) 
                : 0;
        }
        
        if (i >= 2)
        {
            float gap = sc.Low[i] - sc.High[i-2];
            Subgraph_FVG[i] = (gap > Input_FVGThreshold.GetFloat() * sc.TickSize) 
                ? sc.High[i] 
                : ((sc.High[i] - sc.Low[i-2] < -Input_FVGThreshold.GetFloat() * sc.TickSize) 
                    ? sc.Low[i] 
                    : 0);
        }
        
        int longerTimeframeBars = Input_LongerTimeFrame.GetInt() / sc.SecondsPerBar;
        if (i % longerTimeframeBars == 0 && i >= longerTimeframeBars)
        {
            float longerTimeframeReturn = (sc.Close[i] - sc.Close[i - longerTimeframeBars]) / sc.Close[i - longerTimeframeBars];
            Subgraph_LongerTimeframeTrend[i] = longerTimeframeReturn;
        }
        else if (i > 0)
        {
            Subgraph_LongerTimeframeTrend[i] = Subgraph_LongerTimeframeTrend[i - 1];
        }
        
        if (i != sc.ArraySize - 1 || sc.IsFullRecalculation)
            continue;
        
        float deltaDivergence = Subgraph_CumulativeDelta[i] - Subgraph_CumulativeDelta[i-1];
        bool strongBuySignal = deltaDivergence > Input_CumulativeDeltaThreshold.GetFloat() && 
                               Subgraph_VolumeImbalance[i] > 0 && 
                               Subgraph_FVG[i] > 0 &&
                               Subgraph_LongerTimeframeTrend[i] > 0;
        
        bool strongSellSignal = deltaDivergence < -Input_CumulativeDeltaThreshold.GetFloat() && 
                                 Subgraph_VolumeImbalance[i] < 0 && 
                                 Subgraph_FVG[i] < 0 &&
                                 Subgraph_LongerTimeframeTrend[i] < 0;
        
        SCFloatArrayRef volatility = sc.Subgraph[7];
        sc.StdDeviation(sc.Close, volatility, i, 20);
        bool volatilityAcceptable = volatility[i] > Input_VolatilityThreshold.GetFloat();
        
        int currentTime = sc.BaseDateTimeIn[i].GetTimeInSeconds();
        bool withinTradingHours = currentTime >= Input_TradingStartTime.GetTime() && 
                                  currentTime <= Input_TradingEndTime.GetTime();
        
        s_SCPositionData PositionData;
        sc.GetTradePosition(PositionData);
        float accountBalance = PositionData.OpenProfitLoss + PositionData.LastTradeProfitLoss; // Using this as an approximation
        float accountRisk = accountBalance * (Input_RiskPercentage.GetFloat() / 100);
        
        int positionSize = (int)fmin(accountRisk / (Subgraph_ATR[i] * sc.TickSize), (float)Input_MaxPositionSize.GetInt());
        
        if (strongBuySignal && PositionData.PositionQuantity <= 0 && volatilityAcceptable && withinTradingHours)
        {
            s_SCNewOrder NewOrder;
            NewOrder.OrderQuantity = positionSize;
            NewOrder.OrderType = SCT_ORDERTYPE_MARKET;
            NewOrder.TimeInForce = SCT_TIF_GOOD_TILL_CANCELED;
            
            NewOrder.Target1Offset = Input_TakeProfitMultiplier.GetFloat() * Subgraph_ATR[i];
            NewOrder.Stop1Offset = Subgraph_ATR[i];
            
            int Result = sc.BuyEntry(NewOrder);
            if (Result > 0)
            {
                Subgraph_BuyEntry[i] = sc.Low[i];
                
                SCString message;
                message.Format("Buy Entry: Quantity=%d, Price=%.2f, Stop=%.2f, Target=%.2f", 
                               NewOrder.OrderQuantity, sc.Close[i], sc.Close[i] - Subgraph_ATR[i], sc.Close[i] + Input_TakeProfitMultiplier.GetFloat() * Subgraph_ATR[i]);
                sc.AddMessageToLog(message, 0);
            }
            else
            {
                sc.AddMessageToLog(sc.GetTradingErrorTextMessage(Result), 0);
            }
        }
        else if (strongSellSignal && PositionData.PositionQuantity >= 0 && volatilityAcceptable && withinTradingHours)
        {
            s_SCNewOrder NewOrder;
            NewOrder.OrderQuantity = positionSize;
            NewOrder.OrderType = SCT_ORDERTYPE_MARKET;
            NewOrder.TimeInForce = SCT_TIF_GOOD_TILL_CANCELED;
            
            NewOrder.Target1Offset = Input_TakeProfitMultiplier.GetFloat() * Subgraph_ATR[i];
            NewOrder.Stop1Offset = Subgraph_ATR[i];
            
            int Result = sc.SellEntry(NewOrder);
            if (Result > 0)
            {
                Subgraph_SellEntry[i] = sc.High[i];
                
                SCString message;
                message.Format("Sell Entry: Quantity=%d, Price=%.2f, Stop=%.2f, Target=%.2f", 
                               NewOrder.OrderQuantity, sc.Close[i], sc.Close[i] + Subgraph_ATR[i], sc.Close[i] - Input_TakeProfitMultiplier.GetFloat() * Subgraph_ATR[i]);
                sc.AddMessageToLog(message, 0);
            }
            else
            {
                sc.AddMessageToLog(sc.GetTradingErrorTextMessage(Result), 0);
            }
        }
        
        if (PositionData.PositionQuantity != 0)
        {
            int barsHeld = i - sc.GetTradePosition(PositionData);
            if (barsHeld >= Input_MaxHoldingTime.GetInt() * 60 / sc.SecondsPerBar)
            {
                s_SCNewOrder ExitOrder;
                ExitOrder.OrderQuantity = abs(PositionData.PositionQuantity);
                ExitOrder.OrderType = SCT_ORDERTYPE_MARKET;
                
                int Result;
                if (PositionData.PositionQuantity > 0)
                    Result = sc.BuyExit(ExitOrder);
                else
                    Result = sc.SellExit(ExitOrder);
                
                if (Result > 0)
                {
                    SCString message;
                    message.Format("Time-based Exit: Quantity=%d, Price=%.2f", ExitOrder.OrderQuantity, sc.Close[i]);
                    sc.AddMessageToLog(message, 0);
                }
                else
                {
                    sc.AddMessageToLog(sc.GetTradingErrorTextMessage(Result), 0);
                }
            }
        }
    }
    
    // Performance metrics calculation (at the end of each trading day)
    if (sc.Index == sc.ArraySize - 1 && sc.BaseDateTimeIn[sc.Index].GetTime() == HMS_TIME(16, 0, 0))
    {
        float totalReturn = 0;
        float maxDrawdown = 0;
        float peak = 0;
        int tradeCount = 0;
        float sumReturns = 0;
        float sumSquaredReturns = 0;
        
        int TradeCount = sc.GetTradeListSize();
        
        for (int i = 0; i < TradeCount; i++)
        {
            s_ACSTrade Trade;
            sc.GetTradeListEntry(i, Trade);
            
            float tradeReturn = (Trade.ExitPrice - Trade.EntryPrice) / Trade.EntryPrice * (Trade.TradeType == 1 ? 1 : -1);
            totalReturn += tradeReturn;
            sumReturns += tradeReturn;
            sumSquaredReturns += tradeReturn * tradeReturn;
            tradeCount++;
            
            peak = max(peak, totalReturn);
            maxDrawdown = max(maxDrawdown, peak - totalReturn);
        }
        
        if (tradeCount > 0)
        {
            float averageReturn = sumReturns / tradeCount;
            float stdDev = sqrt((sumSquaredReturns - tradeCount * averageReturn * averageReturn) / (tradeCount - 1));
            float sharpeRatio = (averageReturn - 0.02f / 252.0f) / (stdDev * sqrt(252.0f)); // Assuming 2% risk-free rate
            
            float winRate = 0;
            for (int i = 0; i < TradeCount; i++)
            {
                s_ACSTrade Trade;
                sc.GetTradeListEntry(i, Trade);
                if (Trade.TradeProfitLoss > 0)
                    winRate++;
            }
            winRate = (winRate / TradeCount) * 100;
            
            SCString message;
            message.Format("Daily Performance: Return=%.2f%%, MaxDrawdown=%.2f%%, SharpeRatio=%.2f, WinRate=%.2f%%", 
                           totalReturn * 100, maxDrawdown * 100, sharpeRatio, winRate);
            sc.AddMessageToLog(message, 0);
        }
        else
        {
            sc.AddMessageToLog("No trades executed today.", 0);
        }
    }
}