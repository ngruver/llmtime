from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime 
from alpaca_trade_api import REST 
from timedelta import Timedelta 

from models.llmtime import get_llmtime_predictions_data
from finbert_utils import estimate_sentiment

import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


os.environ['MISTRAL_KEY']='iyYT7uLWQZC4jDuVsKSVQ1tMcrZjc6s8'
mistral_client = MistralClient(os.environ['MISTRAL_KEY'])

API_KEY ="PKG9H92SIZ307T5C5YP2"
API_SECRET ="L17qx7T7MvslgQLM76Dzz1eXdui2YE27B7elENpo"
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}

class MLTrader(Strategy): 
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.5,model=None): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.init_model_llm()

    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity
    
    def init_model_llm(self):
        self.model = MistralClient(os.environ['MISTRAL_KEY'])
    def get_sentiments_llm(self,news):
        mistral_sys_message = '''You a trading expert, you analyze news and provide sentiment analysis. the provided sentiment will be one word. The sentiment 
        can be one of three categories: positive, negative, or neutral.do not provide any explanation, just return one word.''' 
        extra_input = ''' Please analyse and summarize the following news articles about SPY index and provide a summurized trading sentiment of SPY index that summurize 
        all of the analysis  based on the information below, just return the one of the three sentiments : positive, negative or neutral,
        do not return any other sentence or word, just return only one word, do not provide explanation\n '''
        for i in range(len(news)):
            extra_input = extra_input + (str(i+1)+news[i]+"\n") 
        response = self.model.chat(
            model='mistral-small',
            messages=[ChatMessage(role="system", content = mistral_sys_message),ChatMessage(role="user", content= (extra_input))]
        )
        return [choice.message.content for choice in response.choices]
    def get_dates(self): 
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment_llm(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        sentiments_llm = self.get_sentiments_llm(news)
        return 1, sentiments_llm[0]

    def get_sentiment(self): 
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        sentiments_llm = self.get_sentiments_llm(news)
        print('\nSentiment LLM :' + str(sentiments_llm))
        probability, sentiment = estimate_sentiment(news)
        print('Sentiment Other : '+sentiment + ' ' + str(probability))
        return probability, sentiment 

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing() 
        probability, sentiment = self.get_sentiment()

        if cash > last_price: 
            if sentiment.lower() == "positive" and probability > .999: 
                if self.last_trade == "sell": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "buy", 
                    type="bracket", 
                    take_profit_price=last_price*1.20, 
                    stop_loss_price=last_price*.95
                )
                self.submit_order(order) 
                self.last_trade = "buy"
            elif sentiment.lower() == "negative" and probability > .999: 
                if self.last_trade == "buy": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "sell", 
                    type="bracket", 
                    take_profit_price=last_price*.8, 
                    stop_loss_price=last_price*1.05
                )
                self.submit_order(order) 
                self.last_trade = "sell"
# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
start_date = datetime(2023,12,1)
end_date = datetime(2023,12,31) 
broker = Alpaca(ALPACA_CREDS) 
strategy = MLTrader(name='mlstrat', broker=broker, 
                    parameters={"symbol":"SPY", 
                                "cash_at_risk":.5},model=mistral_client)
strategy.backtest(
    YahooDataBacktesting, 
    start_date, 
    end_date, 
    parameters={"symbol":"SPY", "cash_at_risk":.5}
)