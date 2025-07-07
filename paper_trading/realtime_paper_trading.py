#!/usr/bin/env python3
# paper_trading/realtime_paper_trading.py - Real-time Paper Trading System
# ระบบเทรด Paper Trading แบบ Real-time

import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import json
import os
from dataclasses import dataclass
from enum import Enum

from utils.config_manager import get_config
from utils.logger import get_logger
from data.data_manager import DataManager

class OrderType(Enum):
    """ประเภทของคำสั่งซื้อขาย"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class Trade:
    """ข้อมูลการเทรด"""
    timestamp: datetime
    symbol: str
    order_type: OrderType
    quantity: float
    price: float
    confidence: float
    portfolio_value_before: float
    portfolio_value_after: float
    profit_loss: float

@dataclass
class Position:
    """ข้อมูลสถานะการถือครอง"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float

class PaperTradingPortfolio:
    """จัดการ Portfolio สำหรับ Paper Trading"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        
    def get_portfolio_value(self) -> float:
        """คำนวณมูลค่า Portfolio ปัจจุบัน"""
        total_value = self.cash
        
        for position in self.positions.values():
            total_value += position.quantity * position.current_price
        
        return total_value
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """ดึงข้อมูล Position ของ symbol"""
        return self.positions.get(symbol)
    
    def execute_trade(self, symbol: str, order_type: OrderType, quantity: float, 
                     price: float, confidence: float) -> bool:
        """ดำเนินการเทรด"""
        try:
            portfolio_value_before = self.get_portfolio_value()
            
            if order_type == OrderType.BUY:
                return self._execute_buy(symbol, quantity, price, confidence, portfolio_value_before)
            elif order_type == OrderType.SELL:
                return self._execute_sell(symbol, quantity, price, confidence, portfolio_value_before)
            
            return False
            
        except Exception as e:
            print(f"Error executing trade: {e}")
            return False
    
    def _execute_buy(self, symbol: str, quantity: float, price: float, 
                    confidence: float, portfolio_value_before: float) -> bool:
        """ดำเนินการซื้อ"""
        total_cost = quantity * price
        
        if total_cost > self.cash:
            return False  # ไม่มีเงินพอ
        
        # หัก Cash
        self.cash -= total_cost
        
        # อัพเดท Position
        if symbol in self.positions:
            # มี Position เดิมอยู่
            position = self.positions[symbol]
            new_quantity = position.quantity + quantity
            new_avg_price = ((position.quantity * position.avg_price) + total_cost) / new_quantity
            
            position.quantity = new_quantity
            position.avg_price = new_avg_price
            position.current_price = price
            position.unrealized_pnl = new_quantity * (price - new_avg_price)
        else:
            # สร้าง Position ใหม่
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
        
        # บันทึก Trade
        portfolio_value_after = self.get_portfolio_value()
        trade = Trade(
            timestamp=datetime.now(),
            symbol=symbol,
            order_type=OrderType.BUY,
            quantity=quantity,
            price=price,
            confidence=confidence,
            portfolio_value_before=portfolio_value_before,
            portfolio_value_after=portfolio_value_after,
            profit_loss=0.0  # ยังไม่มี P&L สำหรับการซื้อ
        )
        self.trades.append(trade)
        
        return True
    
    def _execute_sell(self, symbol: str, quantity: float, price: float, 
                     confidence: float, portfolio_value_before: float) -> bool:
        """ดำเนินการขาย"""
        if symbol not in self.positions:
            return False  # ไม่มี Position
        
        position = self.positions[symbol]
        if position.quantity < quantity:
            return False  # ไม่มีหุ้นพอขาย
        
        # คำนวณ P&L
        profit_loss = quantity * (price - position.avg_price)
        
        # เพิ่ม Cash
        self.cash += quantity * price
        
        # อัพเดท Position
        position.quantity -= quantity
        position.current_price = price
        position.realized_pnl += profit_loss
        
        if position.quantity == 0:
            # ขายหมดแล้ว ลบ Position
            del self.positions[symbol]
        else:
            # ยังมีหุ้นเหลือ
            position.unrealized_pnl = position.quantity * (price - position.avg_price)
        
        # บันทึก Trade
        portfolio_value_after = self.get_portfolio_value()
        trade = Trade(
            timestamp=datetime.now(),
            symbol=symbol,
            order_type=OrderType.SELL,
            quantity=quantity,
            price=price,
            confidence=confidence,
            portfolio_value_before=portfolio_value_before,
            portfolio_value_after=portfolio_value_after,
            profit_loss=profit_loss
        )
        self.trades.append(trade)
        
        return True
    
    def update_positions(self, market_data: Dict[str, float]):
        """อัพเดทราคาปัจจุบันของ Positions"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.current_price = market_data[symbol]
                position.unrealized_pnl = position.quantity * (position.current_price - position.avg_price)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """สรุปข้อมูล Portfolio"""
        portfolio_value = self.get_portfolio_value()
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        
        # คำนวณ metrics
        total_trades = len(self.trades)
        profitable_trades = len([t for t in self.trades if t.profit_loss > 0])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(t.profit_loss for t in self.trades)
        total_loss = sum(t.profit_loss for t in self.trades if t.profit_loss < 0)
        
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'positions': len(self.positions),
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'realized_pnl': sum(p.realized_pnl for p in self.positions.values())
        }

class RealtimeDataSimulator:
    """จำลองข้อมูลแบบ Real-time"""
    
    def __init__(self, historical_data: Dict[str, pd.DataFrame]):
        self.historical_data = historical_data
        self.current_indices = {symbol: 0 for symbol in historical_data.keys()}
        self.data_manager = DataManager()
        
    def get_current_prices(self) -> Dict[str, float]:
        """ดึงราคาปัจจุบัน"""
        current_prices = {}
        
        for symbol, df in self.historical_data.items():
            current_idx = self.current_indices[symbol]
            if current_idx < len(df):
                current_prices[symbol] = df.iloc[current_idx]['close']
                self.current_indices[symbol] += 1
            else:
                # ถ้าข้อมูลหมดแล้ว ใช้ราคาสุดท้าย
                current_prices[symbol] = df.iloc[-1]['close']
        
        return current_prices
    
    def get_current_features(self, symbol: str, lookback_period: int = 60) -> Optional[np.ndarray]:
        """ดึง features ปัจจุบันสำหรับ AI"""
        try:
            current_idx = self.current_indices[symbol]
            if current_idx < lookback_period:
                return None
            
            # ดึงข้อมูล lookback period
            start_idx = current_idx - lookback_period
            end_idx = current_idx
            
            df = self.historical_data[symbol].iloc[start_idx:end_idx]
            
            # สร้าง features
            features_df = self.data_manager.create_features_from_data(df)
            
            # ใช้ข้อมูลล่าสุด
            latest_features = features_df.iloc[-1]
            
            # แปลงเป็น numpy array
            feature_columns = [col for col in features_df.columns 
                             if col not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            features = latest_features[feature_columns].values
            
            return features
            
        except Exception as e:
            print(f"Error getting features for {symbol}: {e}")
            return None
    
    def has_more_data(self) -> bool:
        """ตรวจสอบว่ายังมีข้อมูลเหลืออยู่หรือไม่"""
        for symbol, df in self.historical_data.items():
            if self.current_indices[symbol] < len(df):
                return True
        return False
    
    def reset(self):
        """รีเซ็ต simulator"""
        self.current_indices = {symbol: 0 for symbol in self.historical_data.keys()}

class RealtimePaperTrader:
    """ระบบ Paper Trading แบบ Real-time"""
    
    def __init__(self, ai_model, scaler, config: Dict[str, Any]):
        self.ai_model = ai_model
        self.scaler = scaler
        self.config = config
        self.logger = get_logger()
        
        # ตั้งค่า
        self.initial_capital = config.get('initial_capital', 10000)
        self.max_position_size = config.get('max_position_size', 0.25)
        self.min_confidence = config.get('min_confidence', 0.6)
        self.symbols = config.get('symbols', ['BTC', 'ETH', 'BNB', 'LTC'])
        
        # สร้าง Portfolio
        self.portfolio = PaperTradingPortfolio(self.initial_capital)
        
        # สร้าง Data Manager
        self.data_manager = DataManager()
        
        # สถิติ
        self.start_time = None
        self.total_updates = 0
        self.successful_predictions = 0
        
    async def run_paper_trading(self, duration_minutes: int = 60, 
                               update_interval: int = 5) -> Dict[str, Any]:
        """เรียกใช้ Paper Trading"""
        try:
            self.logger.info(f"Starting paper trading for {duration_minutes} minutes...")
            
            # โหลดข้อมูลย้อนหลัง
            historical_data = self.data_manager.get_combined_data(self.symbols, "data")
            
            if not historical_data:
                self.logger.error("No historical data available")
                return {}
            
            # สร้าง Data Simulator
            data_simulator = RealtimeDataSimulator(historical_data)
            
            # ตั้งค่าเวลา
            self.start_time = datetime.now()
            end_time = self.start_time + timedelta(minutes=duration_minutes)
            
            self.logger.info(f"Paper trading will run until {end_time.strftime('%H:%M:%S')}")
            
            # วนลูปการเทรด
            while datetime.now() < end_time and data_simulator.has_more_data():
                
                # ดึงราคาปัจจุบัน
                current_prices = data_simulator.get_current_prices()
                
                # อัพเดทราคาใน Portfolio
                self.portfolio.update_positions(current_prices)
                
                # ทำการเทรดสำหรับแต่ละ symbol
                for symbol in self.symbols:
                    if symbol in current_prices:
                        await self._process_trading_signal(symbol, current_prices[symbol], data_simulator)
                
                # บันทึกสถิติ
                self._record_portfolio_state(current_prices)
                
                # พัก
                await asyncio.sleep(update_interval)
                
                self.total_updates += 1
                
                # แสดงสถานะทุก 10 updates
                if self.total_updates % 10 == 0:
                    self._log_current_status()
            
            # สรุปผล
            final_results = self._generate_final_results()
            
            self.logger.success("Paper trading completed!")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Paper trading failed: {e}")
            return {}
    
    async def _process_trading_signal(self, symbol: str, current_price: float, 
                                     data_simulator: RealtimeDataSimulator):
        """ประมวลผลสัญญาณการเทรด"""
        try:
            # ดึง features
            features = data_simulator.get_current_features(symbol)
            if features is None:
                return
            
            # ทำนาย
            prediction, confidence = self._get_ai_prediction(features)
            
            if confidence < self.min_confidence:
                return  # ความเชื่อมั่นต่ำเกินไป
            
            # ตัดสินใจเทรด
            if prediction == 2:  # BUY
                await self._execute_buy_signal(symbol, current_price, confidence)
            elif prediction == 0:  # SELL
                await self._execute_sell_signal(symbol, current_price, confidence)
            
            # นับการทำนายที่สำเร็จ
            self.successful_predictions += 1
            
        except Exception as e:
            self.logger.error(f"Error processing trading signal for {symbol}: {e}")
    
    def _get_ai_prediction(self, features: np.ndarray) -> Tuple[int, float]:
        """ดึงการทำนายจาก AI"""
        try:
            # Scale features
            if self.scaler:
                features = self.scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)
            
            # ทำนาย
            if hasattr(self.ai_model, 'predict'):
                prediction = self.ai_model.predict(features)[0]
                
                # ดึง confidence
                if hasattr(self.ai_model, 'predict_proba'):
                    probabilities = self.ai_model.predict_proba(features)[0]
                    confidence = probabilities.max()
                else:
                    confidence = 0.7  # default confidence
                
                return prediction, confidence
            else:
                return 1, 0.0  # HOLD with 0 confidence
                
        except Exception as e:
            self.logger.error(f"Error getting AI prediction: {e}")
            return 1, 0.0
    
    async def _execute_buy_signal(self, symbol: str, price: float, confidence: float):
        """ดำเนินการซื้อ"""
        try:
            # คำนวณจำนวนที่จะซื้อ
            portfolio_value = self.portfolio.get_portfolio_value()
            max_investment = portfolio_value * self.max_position_size
            
            # ปรับตาม confidence
            actual_investment = max_investment * confidence
            
            quantity = actual_investment / price
            
            if quantity > 0 and actual_investment <= self.portfolio.cash:
                success = self.portfolio.execute_trade(
                    symbol, OrderType.BUY, quantity, price, confidence
                )
                
                if success:
                    self.logger.trading_action("BUY", symbol, price, quantity, confidence)
                
        except Exception as e:
            self.logger.error(f"Error executing buy signal: {e}")
    
    async def _execute_sell_signal(self, symbol: str, price: float, confidence: float):
        """ดำเนินการขาย"""
        try:
            position = self.portfolio.get_position(symbol)
            if position and position.quantity > 0:
                
                # ขายบางส่วนตาม confidence
                sell_quantity = position.quantity * confidence
                
                success = self.portfolio.execute_trade(
                    symbol, OrderType.SELL, sell_quantity, price, confidence
                )
                
                if success:
                    self.logger.trading_action("SELL", symbol, price, sell_quantity, confidence)
                
        except Exception as e:
            self.logger.error(f"Error executing sell signal: {e}")
    
    def _record_portfolio_state(self, current_prices: Dict[str, float]):
        """บันทึกสถานะ Portfolio"""
        try:
            portfolio_summary = self.portfolio.get_portfolio_summary()
            
            state = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_summary['portfolio_value'],
                'cash': portfolio_summary['cash'],
                'total_return': portfolio_summary['total_return'],
                'total_trades': portfolio_summary['total_trades'],
                'win_rate': portfolio_summary['win_rate'],
                'current_prices': current_prices.copy(),
                'positions': {symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl
                } for symbol, pos in self.portfolio.positions.items()}
            }
            
            self.portfolio.portfolio_history.append(state)
            
        except Exception as e:
            self.logger.error(f"Error recording portfolio state: {e}")
    
    def _log_current_status(self):
        """แสดงสถานะปัจจุบัน"""
        try:
            summary = self.portfolio.get_portfolio_summary()
            
            self.logger.info(
                f"Status: Portfolio ${summary['portfolio_value']:,.2f} "
                f"({summary['total_return']:+.2f}%) | "
                f"Trades: {summary['total_trades']} | "
                f"Win Rate: {summary['win_rate']:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging status: {e}")
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """สร้างผลลัพธ์สุดท้าย"""
        try:
            summary = self.portfolio.get_portfolio_summary()
            
            # คำนวณเวลาที่ใช้
            duration = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            # คำนวณ metrics เพิ่มเติม
            portfolio_values = [state['portfolio_value'] for state in self.portfolio.portfolio_history]
            
            max_portfolio_value = max(portfolio_values) if portfolio_values else summary['portfolio_value']
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_values)
            
            # สรุปผลการเทรดแต่ละ symbol
            symbol_performance = self._calculate_symbol_performance()
            
            final_results = {
                'initial_capital': self.initial_capital,
                'final_portfolio_value': summary['portfolio_value'],
                'total_return': summary['total_return'],
                'total_trades': summary['total_trades'],
                'win_rate': summary['win_rate'],
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'max_portfolio_value': max_portfolio_value,
                'total_updates': self.total_updates,
                'successful_predictions': self.successful_predictions,
                'prediction_rate': (self.successful_predictions / self.total_updates * 100) if self.total_updates > 0 else 0,
                'duration_minutes': duration.total_seconds() / 60,
                'trades_per_hour': (summary['total_trades'] / (duration.total_seconds() / 3600)) if duration.total_seconds() > 0 else 0,
                'symbol_performance': symbol_performance,
                'portfolio_history': self.portfolio.portfolio_history,
                'all_trades': [self._trade_to_dict(trade) for trade in self.portfolio.trades]
            }
            
            # บันทึกผลลัพธ์
            self._save_results(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error generating final results: {e}")
            return {}
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """คำนวณ Maximum Drawdown"""
        try:
            if not portfolio_values:
                return 0.0
            
            max_value = portfolio_values[0]
            max_drawdown = 0.0
            
            for value in portfolio_values:
                if value > max_value:
                    max_value = value
                
                drawdown = (max_value - value) / max_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return max_drawdown * 100  # เป็นเปอร์เซ็นต์
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, portfolio_values: List[float]) -> float:
        """คำนวณ Sharpe Ratio"""
        try:
            if len(portfolio_values) < 2:
                return 0.0
            
            # คำนวณ returns
            returns = []
            for i in range(1, len(portfolio_values)):
                ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(ret)
            
            if not returns:
                return 0.0
            
            # คำนวณ Sharpe ratio
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # สมมติ risk-free rate = 0
            sharpe_ratio = mean_return / std_return
            
            # Annualize (สมมติว่า update ทุก 5 วินาที)
            periods_per_year = (365 * 24 * 60 * 60) / 5  # updates per year
            annual_sharpe = sharpe_ratio * np.sqrt(periods_per_year)
            
            return annual_sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_symbol_performance(self) -> Dict[str, Dict[str, Any]]:
        """คำนวณผลการเทรดแต่ละ symbol"""
        try:
            symbol_performance = {}
            
            for symbol in self.symbols:
                symbol_trades = [trade for trade in self.portfolio.trades if trade.symbol == symbol]
                
                if symbol_trades:
                    total_trades = len(symbol_trades)
                    profitable_trades = len([t for t in symbol_trades if t.profit_loss > 0])
                    total_profit_loss = sum(t.profit_loss for t in symbol_trades)
                    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                    
                    symbol_performance[symbol] = {
                        'total_trades': total_trades,
                        'profitable_trades': profitable_trades,
                        'win_rate': win_rate,
                        'total_profit_loss': total_profit_loss,
                        'avg_profit_loss': total_profit_loss / total_trades if total_trades > 0 else 0
                    }
                else:
                    symbol_performance[symbol] = {
                        'total_trades': 0,
                        'profitable_trades': 0,
                        'win_rate': 0,
                        'total_profit_loss': 0,
                        'avg_profit_loss': 0
                    }
            
            return symbol_performance
            
        except Exception as e:
            self.logger.error(f"Error calculating symbol performance: {e}")
            return {}
    
    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """แปลง Trade object เป็น dictionary"""
        return {
            'timestamp': trade.timestamp.isoformat(),
            'symbol': trade.symbol,
            'order_type': trade.order_type.value,
            'quantity': trade.quantity,
            'price': trade.price,
            'confidence': trade.confidence,
            'portfolio_value_before': trade.portfolio_value_before,
            'portfolio_value_after': trade.portfolio_value_after,
            'profit_loss': trade.profit_loss
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """บันทึกผลลัพธ์ลงไฟล์"""
        try:
            # สร้าง directory
            os.makedirs('results/paper_trading', exist_ok=True)
            
            # สร้างชื่อไฟล์
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/paper_trading/paper_trade_results_{timestamp}.json"
            
            # แปลง datetime objects เป็น string
            serializable_results = self._make_json_serializable(results)
            
            # บันทึกไฟล์
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.save_session(filename)
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def _make_json_serializable(self, obj):
        """แปลง object ให้ serialize ได้ด้วย JSON"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        else:
            return obj

class PaperTradingManager:
    """จัดการระบบ Paper Trading"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.logger = get_logger()
    
    async def run_comprehensive_paper_trading(self, ai_models: Dict[str, Any], 
                                            scalers: Dict[str, Any]) -> Dict[str, Any]:
        """เรียกใช้ Paper Trading แบบครอบคลุม"""
        try:
            self.logger.info("Starting comprehensive paper trading...")
            
            results = {}
            paper_config = self.config.get_paper_trade_config()
            
            # ทดสอบแต่ละ model
            for model_name, model_data in ai_models.items():
                if isinstance(model_data, tuple):
                    model, metrics = model_data
                elif isinstance(model_data, dict):
                    model = model_data.get('model')
                    metrics = model_data.get('metrics', {})
                else:
                    continue
                
                if model is None:
                    continue
                
                try:
                    self.logger.info(f"Testing {model_name} in paper trading...")
                    
                    # สร้าง Paper Trader
                    trader = RealtimePaperTrader(
                        model, 
                        scalers.get('main'), 
                        paper_config
                    )
                    
                    # เรียกใช้ Paper Trading
                    model_results = await trader.run_paper_trading(
                        duration_minutes=paper_config['duration_minutes'],
                        update_interval=paper_config['update_interval']
                    )
                    
                    results[model_name] = model_results
                    
                    # Log results
                    if model_results:
                        self.logger.paper_trade_result(
                            model_results['initial_capital'],
                            model_results['final_portfolio_value'],
                            model_results['total_return'],
                            model_results['total_trades'],
                            f"{model_results['duration_minutes']:.1f} minutes"
                        )
                    
                except Exception as e:
                    self.logger.error(f"Paper trading failed for {model_name}: {e}")
                    continue
            
            # เปรียบเทียบผลลัพธ์
            comparison = self._compare_paper_trading_results(results)
            results['comparison'] = comparison
            
            self.logger.success("Comprehensive paper trading completed!")
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive paper trading failed: {e}")
            return {}
    
    def _compare_paper_trading_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """เปรียบเทียบผลลัพธ์ Paper Trading"""
        try:
            comparison = {
                'models_tested': [],
                'best_performance': {},
                'metrics_comparison': {}
            }
            
            metrics_to_compare = ['total_return', 'win_rate', 'sharpe_ratio', 'max_drawdown', 'total_trades']
            
            for metric in metrics_to_compare:
                comparison['metrics_comparison'][metric] = {}
            
            best_return = -float('inf')
            best_model = None
            
            for model_name, result in results.items():
                if model_name == 'comparison' or not isinstance(result, dict):
                    continue
                
                comparison['models_tested'].append(model_name)
                
                # เปรียบเทียบ metrics
                for metric in metrics_to_compare:
                    if metric in result:
                        comparison['metrics_comparison'][metric][model_name] = result[metric]
                
                # หา model ที่ดีที่สุด
                total_return = result.get('total_return', -float('inf'))
                if total_return > best_return:
                    best_return = total_return
                    best_model = model_name
            
            if best_model:
                comparison['best_performance'] = {
                    'model': best_model,
                    'total_return': best_return,
                    'details': results[best_model]
                }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing paper trading results: {e}")
            return {}

# Configuration for paper trading
DEFAULT_PAPER_TRADING_CONFIG = {
    'initial_capital': 10000,
    'max_position_size': 0.25,
    'min_confidence': 0.6,
    'symbols': ['BTC', 'ETH', 'BNB', 'LTC'],
    'duration_minutes': 60,
    'update_interval': 5,
    'min_trades_per_hour': 5
}