import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import traceback
import requests
from openai import OpenAI
from datetime import datetime
import numpy as np

# 設置頁面配置
st.header("AI 財報分析系統 ", divider="rainbow")

# 函數：驗證財務數據的完整性和合理性
def validate_financial_data(financial_data):
    """驗證財務數據的完整性和合理性"""
    if not financial_data or len(financial_data) == 0:
        return "錯誤：無財務數據"
    
    required_fields = ['netincomeloss', 'assets', 'revenues', 'stockholdersequity']
    warnings = []
    
    for i, year_data in enumerate(financial_data):
        year = year_data.get('date', f'第{i+1}年')
        
        # 檢查必要欄位
        for field in required_fields:
            if year_data.get(field) is None:
                warnings.append(f"缺少關鍵數據: {field} ({year})")
        
        # 合理性檢查
        assets = year_data.get('assets', 0)
        if assets <= 0:
            warnings.append(f"總資產數據異常: {assets} ({year})")
        
        revenues = year_data.get('revenues', 0)
        if revenues < 0:
            warnings.append(f"營收數據異常: {revenues} ({year})")
    
    if warnings:
        return "數據警告：" + "; ".join(warnings)
    else:
        return "數據驗證通過"

# 函數：分析數據品質
def analyze_data_quality(fmp_data):
    """分析財務數據的品質"""
    quality_report = {
        "數據完整性": "良好",
        "數據年份數": len(fmp_data.get('financial_statements', [])),
        "缺失欄位": [],
        "數據警告": []
    }
    
    financial_data = fmp_data.get('financial_statements', [])
    
    if len(financial_data) < 2:
        quality_report["數據警告"].append("財務數據少於2年，部分趨勢分析可能不準確")
    
    # 檢查關鍵欄位
    critical_fields = ['netincomeloss', 'assets', 'revenues', 'stockholdersequity']
    for field in critical_fields:
        missing_years = []
        for year_data in financial_data:
            if year_data.get(field) is None or year_data.get(field) == 0:
                missing_years.append(year_data.get('date', '未知年份'))
        
        if missing_years:
            quality_report["缺失欄位"].append(f"{field}: {', '.join(missing_years)}")
    
    # 檢查市值數據
    enterprise_data = fmp_data.get('enterprise_values', [])
    if not enterprise_data or not enterprise_data[0].get('marketCapitalization'):
        quality_report["數據警告"].append("缺少市值數據，Altman Z-Score的D項計算可能不準確")
    
    if quality_report["缺失欄位"] or quality_report["數據警告"]:
        quality_report["數據完整性"] = "部分缺失"
    
    return quality_report
def format_large_number(num):
    if num >= 1e12:
        return f"{num/1e12:.2f}兆"
    elif num >= 1e9:
        return f"{num/1e9:.2f}億"
    elif num >= 1e6:
        return f"{num/1e6:.2f}百萬"
    else:
        return f"{num:.2f}"

# 函數：從三個獨立FMP API獲取財務數據
def get_fmp_data_from_three_apis(ticker, api_key):
    """從三個獨立的FMP API獲取財務數據並合併"""
    try:
        # 三個API URLs（更新為 stable 端點與參數）
        income_statement_url = (
            f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&period=annual&limit=5&apikey={api_key}"
        )
        balance_sheet_url = (
            f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={ticker}&period=annual&limit=5&apikey={api_key}"
        )
        cash_flow_url = (
            f"https://financialmodelingprep.com/stable/cash-flow-statement?symbol={ticker}&period=annual&limit=5&apikey={api_key}"
        )
        enterprise_values_url = (
            f"https://financialmodelingprep.com/stable/enterprise-values?symbol={ticker}&apikey={api_key}"
        )
        profile_url = (
            f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key}"
        )
        key_metrics_url = (
            f"https://financialmodelingprep.com/stable/key-metrics-ttm?symbol={ticker}&apikey={api_key}"
        )
        
        # 獲取數據
        st.info(f"正在從FMP API獲取 {ticker} 的財務報表資料...")
        
        # 獲取損益表數據
        income_response = requests.get(income_statement_url)
        if income_response.status_code != 200:
            raise Exception(f"損益表API請求失敗: {income_response.status_code}")
        income_data = income_response.json()
        
        # 獲取資產負債表數據
        balance_response = requests.get(balance_sheet_url)
        if balance_response.status_code != 200:
            raise Exception(f"資產負債表API請求失敗: {balance_response.status_code}")
        balance_data = balance_response.json()
        
        # 獲取現金流量表數據
        cash_flow_response = requests.get(cash_flow_url)
        if cash_flow_response.status_code != 200:
            raise Exception(f"現金流量表API請求失敗: {cash_flow_response.status_code}")
        cash_flow_data = cash_flow_response.json()
        
        # 獲取其他數據
        enterprise_response = requests.get(enterprise_values_url)
        if enterprise_response.status_code != 200:
            raise Exception(f"企業價值API請求失敗: {enterprise_response.status_code}")
        enterprise_data = enterprise_response.json()
        
        profile_response = requests.get(profile_url)
        if profile_response.status_code != 200:
            raise Exception(f"公司資料API請求失敗: {profile_response.status_code}")
        profile_data = profile_response.json()
        
        key_metrics_response = requests.get(key_metrics_url)
        if key_metrics_response.status_code != 200:
            st.warning(f"關鍵指標API請求失敗: {key_metrics_response.status_code}，將嘗試從其他來源獲取PE ratio")
            key_metrics_data = None
        else:
            key_metrics_data = key_metrics_response.json()
        
        # 合併三個財務報表數據
        combined_financial_data = merge_financial_statements(income_data, balance_data, cash_flow_data)
        
        return {
            'financial_statements': combined_financial_data,
            'enterprise_values': enterprise_data,
            'profile': profile_data,
            'key_metrics': key_metrics_data,
            'raw_income': income_data,
            'raw_balance': balance_data,
            'raw_cash_flow': cash_flow_data
        }
        
    except Exception as e:
        raise Exception(f"獲取FMP數據時發生錯誤: {str(e)}")

# 函數：將三個財務報表數據合併為統一格式
def merge_financial_statements(income_data, balance_data, cash_flow_data):
    """將三個獨立的財務報表數據合併為統一格式"""
    try:
        combined_data = []
        
        # 創建日期索引來匹配數據
        dates_income = {item['date']: item for item in income_data}
        dates_balance = {item['date']: item for item in balance_data}
        dates_cash_flow = {item['date']: item for item in cash_flow_data}
        
        # 找出共同的日期
        common_dates = set(dates_income.keys()) & set(dates_balance.keys()) & set(dates_cash_flow.keys())
        
        for date in sorted(common_dates, reverse=True):  # 按日期降序排列
            income_item = dates_income[date]
            balance_item = dates_balance[date]
            cash_flow_item = dates_cash_flow[date]
            
            # 創建統一的數據結構，使用原程式期望的欄位名稱
            merged_item = {
                # 基本信息
                'date': date,
                'symbol': income_item.get('symbol', ''),
                'period': income_item.get('period', ''),
                
                # 損益表數據 - 標準化API到原欄位名稱的映射
                'revenues': income_item.get('revenue', 0),
                'grossprofit': income_item.get('grossProfit', 0),
                'operatingincomeloss': income_item.get('operatingIncome', 0),
                'netincomeloss': income_item.get('netIncome', 0),
                'interestexpensenonoperating': income_item.get('interestExpense', 0),
                'incomelossfromcontinuingoperationsbeforeincometaxes': income_item.get('incomeBeforeTax', 0),
                'weightedaveragenumberofsharesoutstandingbasic': income_item.get('weightedAverageShsOut', 0),
                
                # 資產負債表數據 - 標準化API到原欄位名稱的映射
                'assets': balance_item.get('totalAssets', 0),
                'liabilities': balance_item.get('totalLiabilities', 0),
                'stockholdersequity': balance_item.get('totalStockholdersEquity', 0),
                'assetscurrent': balance_item.get('totalCurrentAssets', 0),
                'liabilitiescurrent': balance_item.get('totalCurrentLiabilities', 0),
                'retainedearningsaccumulateddeficit': balance_item.get('retainedEarnings', 0),
                'longtermdebtnoncurrent': balance_item.get('longTermDebt', 0),
                
                # 現金流量表數據 - 標準化API到原欄位名稱的映射
                'netcashprovidedbyusedinoperatingactivities': (
                    cash_flow_item.get('netCashProvidedByOperatingActivities', 0)
                    if 'netCashProvidedByOperatingActivities' in cash_flow_item
                    else cash_flow_item.get('operatingCashFlow', 0)
                ),
                'netcashprovidedbyusedininvestingactivities': (
                    cash_flow_item.get('netCashProvidedByInvestingActivities', 0)
                    if 'netCashProvidedByInvestingActivities' in cash_flow_item
                    else cash_flow_item.get('netCashUsedForInvestingActivites', 0)
                ),
                'netcashprovidedbyusedinfinancingactivities': (
                    cash_flow_item.get('netCashProvidedByFinancingActivities', 0)
                    if 'netCashProvidedByFinancingActivities' in cash_flow_item
                    else cash_flow_item.get('netCashUsedProvidedByFinancingActivities', 0)
                ),
                'paymentstoacquireproductiveassets': cash_flow_item.get('capitalExpenditure', 0),
                'purchasesofpropertyandequipmentandintangibleassets': cash_flow_item.get('capitalExpenditure', 0)
            }
            
            combined_data.append(merged_item)
        
        return combined_data
        
    except Exception as e:
        raise Exception(f"合併財務數據時發生錯誤: {str(e)}")

# 函數：計算Piotroski F-Score
def calculate_piotroski_fscore(financial_data):
    """計算完整的Piotroski F-Score分析"""
    try:
        if len(financial_data) < 2:
            return {"error": "需要至少2年的財務數據進行F-Score分析"}
        
        current_year = financial_data[0]  # 最新年度
        previous_year = financial_data[1]  # 前一年度
        
        # 獲利能力指標（4項）
        # 1. ROA > 0
        current_roa = current_year.get('netincomeloss', 0) / current_year.get('assets', 1) if current_year.get('assets', 0) != 0 else 0
        roa_positive = 1 if current_roa > 0 else 0
        
        # 2. 營運現金流 > 0
        operating_cf = current_year.get('netcashprovidedbyusedinoperatingactivities', 0)
        cf_positive = 1 if operating_cf > 0 else 0
        
        # 3. ROA年增率 > 0
        previous_roa = previous_year.get('netincomeloss', 0) / previous_year.get('assets', 1) if previous_year.get('assets', 0) != 0 else 0
        roa_improvement = 1 if current_roa > previous_roa else 0
        
        # 4. 營運現金流 > 淨利潤
        net_income = current_year.get('netincomeloss', 0)
        cf_vs_ni = 1 if operating_cf > net_income else 0
        
        # 槓桿與流動性指標（3項）
        # 5. 長期負債比率下降
        current_debt_ratio = current_year.get('longtermdebtnoncurrent', 0) / current_year.get('assets', 1) if current_year.get('assets', 0) != 0 else 0
        previous_debt_ratio = previous_year.get('longtermdebtnoncurrent', 0) / previous_year.get('assets', 1) if previous_year.get('assets', 0) != 0 else 0
        debt_ratio_improvement = 1 if current_debt_ratio < previous_debt_ratio else 0
        
        # 6. 流動比率上升
        current_ratio = current_year.get('assetscurrent', 0) / current_year.get('liabilitiescurrent', 1) if current_year.get('liabilitiescurrent', 0) != 0 else 0
        previous_current_ratio = previous_year.get('assetscurrent', 0) / previous_year.get('liabilitiescurrent', 1) if previous_year.get('liabilitiescurrent', 0) != 0 else 0
        current_ratio_improvement = 1 if current_ratio > previous_current_ratio else 0
        
        # 7. 股份未稀釋
        current_shares = current_year.get('weightedaveragenumberofsharesoutstandingbasic', 0)
        previous_shares = previous_year.get('weightedaveragenumberofsharesoutstandingbasic', 0)
        no_dilution = 1 if current_shares <= previous_shares else 0
        
        # 營運效率指標（2項）
        # 8. 毛利率上升
        current_gross_margin = current_year.get('grossprofit', 0) / current_year.get('revenues', 1) if current_year.get('revenues', 0) != 0 else 0
        previous_gross_margin = previous_year.get('grossprofit', 0) / previous_year.get('revenues', 1) if previous_year.get('revenues', 0) != 0 else 0
        gross_margin_improvement = 1 if current_gross_margin > previous_gross_margin else 0
        
        # 9. 資產周轉率上升
        current_asset_turnover = current_year.get('revenues', 0) / current_year.get('assets', 1) if current_year.get('assets', 0) != 0 else 0
        previous_asset_turnover = previous_year.get('revenues', 0) / previous_year.get('assets', 1) if previous_year.get('assets', 0) != 0 else 0
        asset_turnover_improvement = 1 if current_asset_turnover > previous_asset_turnover else 0
        
        # 計算總分
        total_score = (roa_positive + cf_positive + roa_improvement + cf_vs_ni + 
                      debt_ratio_improvement + current_ratio_improvement + no_dilution +
                      gross_margin_improvement + asset_turnover_improvement)
        
        return {
            'total_score': total_score,
            'profitability_scores': {
                'roa_positive': {'score': roa_positive, 'value': current_roa, 'description': 'ROA > 0'},
                'cf_positive': {'score': cf_positive, 'value': operating_cf, 'description': '營運現金流 > 0'},
                'roa_improvement': {'score': roa_improvement, 'current': current_roa, 'previous': previous_roa, 'description': 'ROA年增率 > 0'},
                'cf_vs_ni': {'score': cf_vs_ni, 'cf': operating_cf, 'ni': net_income, 'description': '營運現金流 > 淨利潤'}
            },
            'leverage_scores': {
                'debt_ratio_improvement': {'score': debt_ratio_improvement, 'current': current_debt_ratio, 'previous': previous_debt_ratio, 'description': '長期負債比率下降'},
                'current_ratio_improvement': {'score': current_ratio_improvement, 'current': current_ratio, 'previous': previous_current_ratio, 'description': '流動比率上升'},
                'no_dilution': {'score': no_dilution, 'current': current_shares, 'previous': previous_shares, 'description': '股份未稀釋'}
            },
            'efficiency_scores': {
                'gross_margin_improvement': {'score': gross_margin_improvement, 'current': current_gross_margin, 'previous': previous_gross_margin, 'description': '毛利率上升'},
                'asset_turnover_improvement': {'score': asset_turnover_improvement, 'current': current_asset_turnover, 'previous': previous_asset_turnover, 'description': '資產周轉率上升'}
            }
        }
        
    except Exception as e:
        return {"error": f"計算Piotroski F-Score時發生錯誤: {str(e)}"}

# 函數：計算Altman Z-Score
def calculate_altman_zscore(financial_data, enterprise_data):
    """計算完整的Altman Z-Score分析"""
    try:
        current_year = financial_data[0]  # 最新年度
        
        # 獲取市值數據
        market_cap = 0
        if enterprise_data and len(enterprise_data) > 0:
            market_cap = enterprise_data[0].get('marketCapitalization', 0)
        
        # A項：營運資本/總資產
        working_capital = current_year.get('assetscurrent', 0) - current_year.get('liabilitiescurrent', 0)
        total_assets = current_year.get('assets', 0)
        a_ratio = working_capital / total_assets if total_assets != 0 else 0
        a_component = 1.2 * a_ratio
        
        # B項：保留盈餘/總資產
        retained_earnings = current_year.get('retainedearningsaccumulateddeficit', 0)
        b_ratio = retained_earnings / total_assets if total_assets != 0 else 0
        b_component = 1.4 * b_ratio
        
        # C項：EBIT/總資產
        operating_income = current_year.get('operatingincomeloss', 0)
        interest_expense = current_year.get('interestexpensenonoperating', 0)
        ebit = operating_income + interest_expense  # 修正：不使用abs()，保持利息費用的原始符號
        c_ratio = ebit / total_assets if total_assets != 0 else 0
        c_component = 3.3 * c_ratio
        
        # D項：市值/總負債
        total_liabilities = current_year.get('liabilities', 0)
        d_ratio = market_cap / total_liabilities if total_liabilities != 0 else 0
        d_component = 0.6 * d_ratio
        
        # E項：營收/總資產
        revenues = current_year.get('revenues', 0)
        e_ratio = revenues / total_assets if total_assets != 0 else 0
        e_component = 1.0 * e_ratio
        
        # 最終Z-Score
        z_score = a_component + b_component + c_component + d_component + e_component
        
        # 風險等級判斷
        if z_score > 2.99:
            risk_level = "安全區域"
        elif z_score > 1.81:
            risk_level = "灰色區域"
        else:
            risk_level = "危險區域"
        
        return {
            'z_score': z_score,
            'risk_level': risk_level,
            'components': {
                'a_component': {'value': a_component, 'ratio': a_ratio, 'description': '營運資本/總資產 × 1.2'},
                'b_component': {'value': b_component, 'ratio': b_ratio, 'description': '保留盈餘/總資產 × 1.4'},
                'c_component': {'value': c_component, 'ratio': c_ratio, 'description': 'EBIT/總資產 × 3.3'},
                'd_component': {'value': d_component, 'ratio': d_ratio, 'description': '市值/總負債 × 0.6'},
                'e_component': {'value': e_component, 'ratio': e_ratio, 'description': '營收/總資產 × 1.0'}
            },
            'detailed_data': {
                'working_capital': working_capital,
                'total_assets': total_assets,
                'retained_earnings': retained_earnings,
                'ebit': ebit,
                'market_cap': market_cap,
                'total_liabilities': total_liabilities,
                'revenues': revenues
            }
        }
        
    except Exception as e:
        return {"error": f"計算Altman Z-Score時發生錯誤: {str(e)}"}

# 函數：計算杜邦分析
def calculate_dupont_analysis(financial_data):
    """計算完整的杜邦分析"""
    try:
        dupont_results = []
        
        for year_data in financial_data[:3]:  # 分析最近3年
            net_income = year_data.get('netincomeloss', 0)
            revenues = year_data.get('revenues', 0)
            total_assets = year_data.get('assets', 0)
            stockholders_equity = year_data.get('stockholdersequity', 0)
            
            # 計算三個因子
            net_margin = net_income / revenues if revenues != 0 else 0
            asset_turnover = revenues / total_assets if total_assets != 0 else 0
            equity_multiplier = total_assets / stockholders_equity if stockholders_equity != 0 else 0
            
            # 計算ROE
            roe_calculated = net_margin * asset_turnover * equity_multiplier
            roe_direct = net_income / stockholders_equity if stockholders_equity != 0 else 0
            
            dupont_results.append({
                'date': year_data.get('date'),
                'net_margin': net_margin,
                'asset_turnover': asset_turnover,
                'equity_multiplier': equity_multiplier,
                'roe_calculated': roe_calculated,
                'roe_direct': roe_direct,
                'components': {
                    'net_income': net_income,
                    'revenues': revenues,
                    'total_assets': total_assets,
                    'stockholders_equity': stockholders_equity
                }
            })
        
        return {
            'yearly_analysis': dupont_results,
            'trend_analysis': analyze_dupont_trends(dupont_results)
        }
        
    except Exception as e:
        return {"error": f"計算杜邦分析時發生錯誤: {str(e)}"}

def analyze_dupont_trends(dupont_results):
    """分析杜邦三因子的趨勢變化"""
    if len(dupont_results) < 2:
        return {"error": "需要至少2年數據進行趨勢分析"}
    
    current = dupont_results[0]
    previous = dupont_results[1]
    
    return {
        'net_margin_change': current['net_margin'] - previous['net_margin'],
        'asset_turnover_change': current['asset_turnover'] - previous['asset_turnover'],
        'equity_multiplier_change': current['equity_multiplier'] - previous['equity_multiplier'],
        'roe_change': current['roe_direct'] - previous['roe_direct'],
        'primary_driver': determine_primary_driver(current, previous)
    }

def determine_primary_driver(current, previous):
    """判斷ROE變化的主要驅動因子"""
    changes = {
        'net_margin': abs(current['net_margin'] - previous['net_margin']),
        'asset_turnover': abs(current['asset_turnover'] - previous['asset_turnover']),
        'equity_multiplier': abs(current['equity_multiplier'] - previous['equity_multiplier'])
    }
    return max(changes, key=changes.get)

# 函數：計算現金流分析
def calculate_cashflow_analysis(financial_data):
    """計算完整的現金流分析"""
    try:
        current_year = financial_data[0]
        
        operating_cf = current_year.get('netcashprovidedbyusedinoperatingactivities', 0)
        investing_cf = current_year.get('netcashprovidedbyusedininvestingactivities', 0)
        financing_cf = current_year.get('netcashprovidedbyusedinfinancingactivities', 0)
        net_income = current_year.get('netincomeloss', 0)
        capex = current_year.get('paymentstoacquireproductiveassets', 0)
        
        # 1. 營運現金流品質
        cf_quality = operating_cf / net_income if net_income != 0 else 0
        
        # 2. 自由現金流 - 修正：明確處理資本支出
        free_cash_flow = operating_cf - abs(capex)  # 明確減去資本支出的絕對值
        
        # 3. 現金流結構分析
        total_cf = operating_cf + investing_cf + financing_cf
        
        structure_analysis = {
            'operating_percentage': (operating_cf / total_cf * 100) if total_cf != 0 else 0,
            'investing_percentage': (investing_cf / total_cf * 100) if total_cf != 0 else 0,
            'financing_percentage': (financing_cf / total_cf * 100) if total_cf != 0 else 0
        }
        
        # 現金流品質評估
        if cf_quality >= 1.2:
            quality_assessment = "優秀"
        elif cf_quality >= 1.0:
            quality_assessment = "良好"
        elif cf_quality >= 0.8:
            quality_assessment = "尚可"
        else:
            quality_assessment = "需關注"
        
        return {
            'cf_quality_ratio': cf_quality,
            'free_cash_flow': free_cash_flow,
            'quality_assessment': quality_assessment,
            'structure_analysis': structure_analysis,
            'detailed_flows': {
                'operating_cf': operating_cf,
                'investing_cf': investing_cf,
                'financing_cf': financing_cf,
                'net_income': net_income,
                'capex': capex,
                'total_cf': total_cf
            }
        }
        
    except Exception as e:
        return {"error": f"計算現金流分析時發生錯誤: {str(e)}"}

# 函數：獲取PE ratio
def get_pe_ratio(profile_data, key_metrics_data, show_debug=False):
    """從不同來源獲取PE ratio"""
    pe_ratio = None
    
    # 如果開啟調試模式，顯示可用的欄位
    if show_debug and profile_data:
        st.write("Profile API 可用欄位：", list(profile_data.keys()))
        if key_metrics_data:
            st.write("Key Metrics API 可用欄位：", list(key_metrics_data[0].keys()) if key_metrics_data else "無數據")
    
    # 嘗試從profile API獲取PE ratio
    possible_pe_fields = ['pe', 'peRatio', 'pe_ratio', 'peRatioTTM', 'priceEarningsRatio']
    for field in possible_pe_fields:
        if field in profile_data and profile_data[field] is not None:
            pe_ratio = profile_data[field]
            if show_debug:
                st.write(f"找到PE ratio在profile API: {field} = {pe_ratio}")
            break
    
    # 如果profile API中沒有找到，嘗試從key metrics API獲取
    if pe_ratio is None and key_metrics_data and len(key_metrics_data) > 0:
        key_metrics = key_metrics_data[0]
        possible_pe_fields_km = ['peRatioTTM', 'pe_ratio', 'peRatio', 'priceEarningsRatio']
        for field in possible_pe_fields_km:
            if field in key_metrics and key_metrics[field] is not None:
                pe_ratio = key_metrics[field]
                if show_debug:
                    st.write(f"找到PE ratio在key metrics API: {field} = {pe_ratio}")
                break
    
    return pe_ratio if pe_ratio is not None else 'N/A'

def prepare_comprehensive_analysis_data(fmp_data, ticker):
    """準備完整的分析數據，包含計算結果、原始數據和數據品質報告"""
    try:
        # 數據品質分析
        data_quality = analyze_data_quality(fmp_data)
        
        # 財務數據驗證
        validation_result = validate_financial_data(fmp_data['financial_statements'])
        
        # 計算各項分析指標
        piotroski_results = calculate_piotroski_fscore(fmp_data['financial_statements'])
        zscore_results = calculate_altman_zscore(fmp_data['financial_statements'], fmp_data['enterprise_values'])
        dupont_results = calculate_dupont_analysis(fmp_data['financial_statements'])
        cashflow_results = calculate_cashflow_analysis(fmp_data['financial_statements'])
        
        # 生成分析限制說明
        analysis_limitations = []
        if "數據警告" in validation_result:
            analysis_limitations.append("部分財務數據缺失，可能影響分析準確性")
        if data_quality["數據年份數"] < 3:
            analysis_limitations.append("財務數據年份較少，趨勢分析可能不夠全面")
        if data_quality["缺失欄位"]:
            analysis_limitations.append("部分關鍵財務指標缺失，請注意相關分析結果")
        
        # 組織完整的分析數據
        comprehensive_data = {
            "公司名稱": ticker.upper(),
            "分析期間": "最近5年年度報表",
            "數據品質報告": {
                "驗證結果": validation_result,
                "品質分析": data_quality,
                "分析限制": analysis_limitations if analysis_limitations else ["無特殊限制"]
            },
            "計算結果": {
                "piotroski_fscore": piotroski_results,
                "altman_zscore": zscore_results,
                "dupont_analysis": dupont_results,
                "cashflow_analysis": cashflow_results
            },
            "完整財務數據": {
                "財務報表數據": fmp_data['financial_statements'],
                "企業價值數據": fmp_data['enterprise_values'],
                "公司基本資料": fmp_data['profile'],
                "關鍵指標數據": fmp_data['key_metrics']
            }
        }
        
        return json.dumps(comprehensive_data, ensure_ascii=False, indent=2)
    except Exception as e:
        raise Exception(f"準備分析數據時發生錯誤: {str(e)}")

# 函數：使用OpenAI分析財務數據
def analyze_with_openai(comprehensive_data, api_key, ticker):
    try:
        import httpx
        # 強制不使用代理，解決 proxies 參數報錯問題
        client = OpenAI(
            api_key=api_key,
            http_client=httpx.Client(proxies={})
        )
        # System 角色：設定 AI 的專業角色與語氣
        system_message = {
            "role": "system",
            "content": "你是一位專業的財務分析師，精通財報分析和投資評估"
        }

        # User 角色：使用優化後的prompt
        user_prompt = f"""
### 已計算完成的標準化財務指標

#### Piotroski F-Score 分析結果
請基於以下精確計算的F-Score結果進行分析解讀

#### Altman Z-Score 分析結果
請基於以下精確計算的Z-Score結果進行風險評估

#### 杜邦分析計算結果
請基於以下精確計算的ROE三因子結果進行趨勢分析

#### 現金流分析計算結果
請基於以下精確計算的現金流比率結果進行品質分析

### 完整財務報表數據（供深度分析）

{comprehensive_data}

### 分析要求
請基於已計算結果進行標準分析，同時利用完整數據進行以下深度分析：

**重要提醒：請特別注意數據品質報告中提到的限制，在分析時適當說明數據完整性對結果的影響**

#### 1. Piotroski F-Score 解讀
- 使用已計算的得分，解讀各項指標的投資意義
- 分析得分背後的業務狀況

#### 2. Altman Z-Score 風險評估
- 基於已計算的Z-Score值進行風險等級判斷
- 分析各組成要素對整體風險的影響

#### 3. 杜邦分析趨勢洞察
- 基於精確的ROE數值，分析三因子的趨勢變化
- 識別影響ROE變化的主要驅動力
- 比較同業或歷史水準的競爭優勢
- 發現財務效率的改善或惡化跡象

#### 4. 現金流結構深度分析
- 基於精確的現金流比率，分析現金流品質
- 識別營運現金流的組成變化和可持續性
- 分析資本支出模式和投資策略
- 發現現金流與獲利品質的一致性

#### 5. 綜合財務健康診斷
- 結合四項分析發現潛在的財務風險或機會
- 識別可能被忽略的財務異常或優勢
- 提供超越單一指標的整體評估
- 發現報表數字間的關聯性和矛盾點

### 綜合評估要求
#### 四階段評分總結表格
| 分析階段 | 評分/狀態 | 評價 | 主要發現 |
|----------|-----------|------|----------|
| Piotroski F-Score | X/9分 | 優秀/良好/一般 | ... |
| Altman Z-Score | X.XX | 安全/灰色/危險 | ... |
| 杜邦分析 | ROE X% | 卓越/良好/一般 | ... |
| 現金流分析 | 評價 | 優秀/良好/需關注 | ... |

#### 分析結論
基於四階段分析結果，提供：
- **主要優勢**：列出3-5個關鍵優勢
- **風險因素**：列出需要關注的風險點
- **後續追蹤重點**：投資後需要監控的關鍵指標
- **財報綜合評比**：請使用三大財報綜合分析該公司「營運績效分析」、「財務結構分析」、「現金流量分析」與「總結分析」

#### 分析要求
- 所有結論都要有數據支撐
- 使用表格整理複雜數據
- 如果資料缺漏請提出，不要生成假資料計算

分析目標公司：{ticker}
分析期間：最近5年年度報表"""

        # 呼叫 OpenAI API
        completion = client.chat.completions.create(
            model="o4-mini",
            messages=[
                system_message,
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return completion.choices[0].message.content        
    except Exception as e:
        stack_trace = traceback.format_exc()
        return f"分析數據時發生錯誤：\n{str(e)}\n\n詳細錯誤訊息：\n{stack_trace}"

# 函數：處理FMP財務數據用於展示
def process_financial_data_for_display(financial_data):
    """處理FMP財務數據，轉換為適合展示的格式"""
    try:
        if not financial_data or len(financial_data) == 0:
            return None, None, None
        
        # 創建DataFrames
        income_data = []
        balance_data = []
        cash_flow_data = []
        
        for year_data in financial_data:
            date = year_data.get('date', '')
            
            # 損益表數據
            income_row = {
                'Date': date,
                'Total Revenue': year_data.get('revenues', 0),
                'Gross Profit': year_data.get('grossprofit', 0),
                'Operating Income': year_data.get('operatingincomeloss', 0),
                'Net Income': year_data.get('netincomeloss', 0)
            }
            income_data.append(income_row)
            
            # 資產負債表數據
            balance_row = {
                'Date': date,
                'Total Assets': year_data.get('assets', 0),
                'Total Liabilities': year_data.get('liabilities', 0),
                'Total Stockholder Equity': year_data.get('stockholdersequity', 0),
                'Current Assets': year_data.get('assetscurrent', 0),
                'Current Liabilities': year_data.get('liabilitiescurrent', 0)
            }
            balance_data.append(balance_row)
            
            # 現金流量表數據
            cash_row = {
                'Date': date,
                'Operating Cash Flow': year_data.get('netcashprovidedbyusedinoperatingactivities', 0),
                'Investing Cash Flow': year_data.get('netcashprovidedbyusedininvestingactivities', 0),
                'Financing Cash Flow': year_data.get('netcashprovidedbyusedinfinancingactivities', 0),
                'Capital Expenditures': year_data.get('paymentstoacquireproductiveassets', 0)
            }
            cash_flow_data.append(cash_row)
        
        # 轉換為DataFrame並設置索引
        income_df = pd.DataFrame(income_data).set_index('Date')
        balance_df = pd.DataFrame(balance_data).set_index('Date')
        cash_df = pd.DataFrame(cash_flow_data).set_index('Date')
        
        # 反轉順序以使最新數據在前
        income_df = income_df.iloc[::-1]
        balance_df = balance_df.iloc[::-1]
        cash_df = cash_df.iloc[::-1]
        
        return income_df, balance_df, cash_df
        
    except Exception as e:
        raise Exception(f"處理財務數據時發生錯誤: {str(e)}")

# 主應用程式
def main():
    
    # 側邊欄
    st.sidebar.header("Code Gym", divider="rainbow")
    
    # 股票代碼輸入
    ticker = st.sidebar.text_input("輸入股票代碼（例如：NVDA 代表NVIDIA）", "NVDA")
    
    # FMP API金鑰
    fmp_api_key = st.sidebar.text_input("輸入資料API Key", type="password", value="")
    
    # OpenAI API金鑰
    openai_api_key = st.sidebar.text_input("輸入OpenAI API金鑰", type="password", value="")
    
    # 按鈕來執行分析
    if st.sidebar.button("分析股票"):
        if not ticker:
            st.warning("請輸入股票代碼")
            return
        
        if not fmp_api_key:
            st.warning("請輸入FMP API金鑰")
            return
        
        try:
            # 從三個獨立的FMP API獲取數據
            fmp_data = get_fmp_data_from_three_apis(ticker, fmp_api_key)
            
            # 顯示基本資訊
            try:
                profile_data = fmp_data['profile'][0] if fmp_data['profile'] else {}
                
                # 相容不同欄位命名（profile API 在 stable 端點可能不同鍵名）
                company_name = profile_data.get('companyName') or profile_data.get('companyname') or profile_data.get('company') or ticker
                sector = profile_data.get('sector') or profile_data.get('industry') or 'N/A'
                industry = profile_data.get('industry') or profile_data.get('subIndustry') or 'N/A'
                key_metrics_data = fmp_data['key_metrics']
                
                # 獲取PE ratio
                pe_ratio = get_pe_ratio(profile_data, key_metrics_data, False)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader(f"{company_name}")
                    st.write(f"**產業類別:** {sector}")
                    st.write(f"**行業:** {industry}")
                
                with col2:
                    current_price = profile_data.get('price') or profile_data.get('priceCurrent') or profile_data.get('currentPrice') or 0
                    price_change = profile_data.get('changes') or profile_data.get('change') or profile_data.get('changePercentage') or 0
                    st.metric("當前價格", f"${current_price}", f"{price_change:.2f}")
                
                with col3:
                    market_cap = profile_data.get('mktCap') or profile_data.get('marketCap') or profile_data.get('marketCapitalization') or 0
                    st.metric("市值", format_large_number(market_cap), f"本益比: {pe_ratio}")
                    
            except Exception as e:
                st.error(f"獲取基本資訊時發生錯誤：{str(e)}")
            
            # 處理財務數據
            try:
                # 驗證財務數據
                validation_result = validate_financial_data(fmp_data['financial_statements'])
                if "錯誤" in validation_result:
                    st.error(validation_result)
                    return
                elif "數據警告" in validation_result:
                    st.warning(validation_result)
                
                income_df, balance_df, cash_df = process_financial_data_for_display(fmp_data['financial_statements'])
                
                if income_df is None:
                    st.error("無法獲取有效的財務數據")
                    return
                                       
            except Exception as e:
                st.error(f"處理財報資料時發生錯誤：{str(e)}")
                return
            
            # 創建財報分析和AI分析標籤
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["損益表分析", "資產負債表分析", "現金流量表分析", "四階段財報分析", "AI分析"])
            
            with tab1:
                st.subheader("損益表分析")
                
                try:
                    if not income_df.empty:
                        # 顯示關鍵指標圖表
                        st.subheader("關鍵指標")
                        
                        fig = go.Figure()
                        
                        for column in income_df.columns:
                            fig.add_trace(go.Bar(
                                x=income_df.index,
                                y=income_df[column],
                                name=column
                            ))
                        
                        fig.update_layout(
                            title="關鍵財務指標",
                            xaxis_title="日期",
                            yaxis_title="金額",
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 顯示完整損益表
                        st.subheader("完整損益表")
                        st.dataframe(income_df, use_container_width=True)
                    else:
                        st.write("沒有可用的損益表資料")
                        
                except Exception as e:
                    st.error(f"損益表分析時發生錯誤：{str(e)}")
            
            with tab2:
                st.subheader("資產負債表分析")
                
                try:
                    if not balance_df.empty:
                        # 顯示關鍵指標圖表
                        st.subheader("關鍵指標")
                        
                        fig = go.Figure()
                        
                        for column in balance_df.columns:
                            fig.add_trace(go.Bar(
                                x=balance_df.index,
                                y=balance_df[column],
                                name=column
                            ))
                        
                        fig.update_layout(
                            title="資產負債關鍵指標",
                            xaxis_title="日期",
                            yaxis_title="金額",
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 計算財務比率
                        balance_df_copy = balance_df.copy()
                        
                        if 'Total Assets' in balance_df.columns and 'Total Liabilities' in balance_df.columns:
                            balance_df_copy["負債比率 (%)"] = (balance_df_copy['Total Liabilities'] / balance_df_copy['Total Assets'] * 100).round(2)
                        
                        if 'Current Assets' in balance_df.columns and 'Current Liabilities' in balance_df.columns:
                            balance_df_copy["流動比率"] = (balance_df_copy['Current Assets'] / balance_df_copy['Current Liabilities']).round(2)
                        
                        # 顯示財務比率
                        ratio_columns = [col for col in balance_df_copy.columns if '比率' in col]
                        if ratio_columns:
                            st.subheader("財務比率")
                            st.dataframe(balance_df_copy[ratio_columns], use_container_width=True)
                            
                            # 繪製財務比率圖表
                            fig_ratios = go.Figure()
                            
                            for column in ratio_columns:
                                fig_ratios.add_trace(go.Scatter(
                                    x=balance_df_copy.index,
                                    y=balance_df_copy[column],
                                    name=column,
                                    mode='lines+markers'
                                ))
                            
                            fig_ratios.update_layout(
                                title="財務比率趨勢",
                                xaxis_title="日期",
                                yaxis_title="比率"
                            )
                            
                            st.plotly_chart(fig_ratios, use_container_width=True)
                        
                        # 顯示完整資產負債表
                        st.subheader("完整資產負債表")
                        st.dataframe(balance_df, use_container_width=True)
                    else:
                        st.write("沒有可用的資產負債表資料")
                        
                except Exception as e:
                    st.error(f"資產負債表分析時發生錯誤：{str(e)}")
            
            with tab3:
                st.subheader("現金流量表分析")
                
                try:
                    if not cash_df.empty:
                        # 顯示關鍵指標圖表（移除Capital Expenditures）
                        st.subheader("關鍵指標")
                        
                        # 只顯示現金流量指標，排除Capital Expenditures
                        cash_flow_columns = [col for col in cash_df.columns if col != 'Capital Expenditures']
                        
                        if cash_flow_columns:
                            fig = go.Figure()
                            
                            for column in cash_flow_columns:
                                fig.add_trace(go.Bar(
                                    x=cash_df.index,
                                    y=cash_df[column],
                                    name=column
                                ))
                            
                            fig.update_layout(
                                title="現金流量關鍵指標",
                                xaxis_title="日期",
                                yaxis_title="金額",
                                barmode='group'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 計算自由現金流
                        if 'Operating Cash Flow' in cash_df.columns and 'Capital Expenditures' in cash_df.columns:
                            cash_df_copy = cash_df.copy()
                            # 修正：明確處理資本支出計算
                            cash_df_copy["Free Cash Flow"] = cash_df_copy['Operating Cash Flow'] - cash_df_copy['Capital Expenditures'].abs()
                            
                            fig_fcf = go.Figure()
                            fig_fcf.add_trace(go.Scatter(
                                x=cash_df_copy.index,
                                y=cash_df_copy["Free Cash Flow"],
                                mode='lines+markers',
                                name='Free Cash Flow',
                                line=dict(color='#1f77b4', width=3),
                                marker=dict(size=8)
                            ))
                            
                            st.plotly_chart(fig_fcf, use_container_width=True)
                        
                        # 顯示完整現金流量表
                        st.subheader("完整現金流量表")
                        st.dataframe(cash_df, use_container_width=True)
                    else:
                        st.write("沒有可用的現金流量表資料")
                        
                except Exception as e:
                    st.error(f"現金流量表分析時發生錯誤：{str(e)}")
            
            with tab4:
                st.subheader("四階段財報分析")
                
                try:
                    # 先進行數據驗證
                    validation_result = validate_financial_data(fmp_data['financial_statements'])
                    data_quality = analyze_data_quality(fmp_data)
                    
                    # 顯示數據品質資訊
                    if "數據警告" in validation_result or data_quality["數據完整性"] != "良好":
                        with st.expander("📊 數據品質報告", expanded=False):
                            st.write("**數據驗證結果:**", validation_result)
                            st.write("**數據完整性:**", data_quality["數據完整性"])
                            if data_quality["缺失欄位"]:
                                st.write("**缺失欄位:**", "; ".join(data_quality["缺失欄位"]))
                            if data_quality["數據警告"]:
                                st.write("**數據警告:**", "; ".join(data_quality["數據警告"]))
                    
                    # 計算各項分析指標
                    piotroski_results = calculate_piotroski_fscore(fmp_data['financial_statements'])
                    zscore_results = calculate_altman_zscore(fmp_data['financial_statements'], fmp_data['enterprise_values'])
                    dupont_results = calculate_dupont_analysis(fmp_data['financial_statements'])
                    cashflow_results = calculate_cashflow_analysis(fmp_data['financial_statements'])
                    
                    # 階段一：Piotroski F-Score 分析表格
                    st.subheader("📊 階段一：Piotroski F-Score 分析")
                    
                    if "error" not in piotroski_results:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 獲利能力指標表格
                            st.markdown("**獲利能力指標（4項）**")
                            profitability_data = []
                            for key, value in piotroski_results['profitability_scores'].items():
                                profitability_data.append({
                                    "指標": value['description'],
                                    "當前值": f"{value.get('value', value.get('current', 0)):.4f}",
                                    "前期值": f"{value.get('previous', 'N/A')}" if 'previous' in value else "N/A",
                                    "得分": value['score'],
                                    "狀態": "✅ 通過" if value['score'] == 1 else "❌ 未通過"
                                })
                            st.dataframe(pd.DataFrame(profitability_data), use_container_width=True, hide_index=True)
                            
                            # 槓桿與流動性指標表格
                            st.markdown("**槓桿與流動性指標（3項）**")
                            leverage_data = []
                            for key, value in piotroski_results['leverage_scores'].items():
                                leverage_data.append({
                                    "指標": value['description'],
                                    "當前值": f"{value.get('current', 0):.4f}",
                                    "前期值": f"{value.get('previous', 0):.4f}",
                                    "得分": value['score'],
                                    "狀態": "✅ 通過" if value['score'] == 1 else "❌ 未通過"
                                })
                            st.dataframe(pd.DataFrame(leverage_data), use_container_width=True, hide_index=True)
                            
                            # 營運效率指標表格
                            st.markdown("**營運效率指標（2項）**")
                            efficiency_data = []
                            for key, value in piotroski_results['efficiency_scores'].items():
                                efficiency_data.append({
                                    "指標": value['description'],
                                    "當前值": f"{value.get('current', 0):.4f}",
                                    "前期值": f"{value.get('previous', 0):.4f}",
                                    "得分": value['score'],
                                    "狀態": "✅ 通過" if value['score'] == 1 else "❌ 未通過"
                                })
                            st.dataframe(pd.DataFrame(efficiency_data), use_container_width=True, hide_index=True)
                        
                        with col2:
                            # F-Score 總分顯示
                            total_score = piotroski_results['total_score']
                            st.metric("Piotroski F-Score 總分", f"{total_score}/9", 
                                     delta=f"{'優秀' if total_score >= 7 else '良好' if total_score >= 5 else '一般' if total_score >= 3 else '需改善'}")
                            
                            # 分數分佈圓餅圖
                            pass_count = total_score
                            fail_count = 9 - total_score
                            
                            import plotly.express as px
                            fig_pie = px.pie(
                                values=[pass_count, fail_count],
                                names=["通過", "未通過"],
                                title="F-Score 指標通過率",
                                color_discrete_sequence=["#2E8B57", "#DC143C"]
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.error(f"Piotroski F-Score 計算錯誤: {piotroski_results['error']}")
                    
                    st.divider()
                    
                    # 階段二：Altman Z-Score 分析表格
                    st.subheader("📊 階段二：Altman Z-Score 分析")
                    
                    if "error" not in zscore_results:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Z-Score 組成要素表格
                            zscore_components_data = []
                            for key, value in zscore_results['components'].items():
                                zscore_components_data.append({
                                    "組成要素": value['description'],
                                    "比率值": f"{value['ratio']:.4f}",
                                    "權重後數值": f"{value['value']:.4f}"
                                })
                            st.dataframe(pd.DataFrame(zscore_components_data), use_container_width=True, hide_index=True)
                            
                            # 詳細數據表格
                            st.markdown("**計算基礎數據**")
                            detailed_data = zscore_results['detailed_data']
                            zscore_detail_data = [
                                {"項目": "營運資本", "數值": f"{detailed_data['working_capital']:,.0f}"},
                                {"項目": "總資產", "數值": f"{detailed_data['total_assets']:,.0f}"},
                                {"項目": "保留盈餘", "數值": f"{detailed_data['retained_earnings']:,.0f}"},
                                {"項目": "EBIT", "數值": f"{detailed_data['ebit']:,.0f}"},
                                {"項目": "市值", "數值": f"{detailed_data['market_cap']:,.0f}"},
                                {"項目": "總負債", "數值": f"{detailed_data['total_liabilities']:,.0f}"},
                                {"項目": "營收", "數值": f"{detailed_data['revenues']:,.0f}"}
                            ]
                            st.dataframe(pd.DataFrame(zscore_detail_data), use_container_width=True, hide_index=True)
                        
                        with col2:
                            # Z-Score 結果顯示
                            z_score = zscore_results['z_score']
                            risk_level = zscore_results['risk_level']
                            
                            # 根據風險等級設定顏色
                            if risk_level == "安全區域":
                                delta_color = "normal"
                                risk_emoji = "🟢"
                            elif risk_level == "灰色區域":
                                delta_color = "off"
                                risk_emoji = "🟡"
                            else:
                                delta_color = "inverse"
                                risk_emoji = "🔴"
                            
                            st.metric("Altman Z-Score", f"{z_score:.2f}", 
                                     delta=f"{risk_emoji} {risk_level}")
                            
                            # Z-Score 區間圖表
                            fig_gauge = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = z_score,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Z-Score"},
                                gauge = {
                                    'axis': {'range': [0, 5]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 1.81], 'color': "lightcoral"},
                                        {'range': [1.81, 2.99], 'color': "lightyellow"},
                                        {'range': [2.99, 5], 'color': "lightgreen"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': z_score
                                    }
                                }
                            ))
                            fig_gauge.update_layout(height=250)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                    else:
                        st.error(f"Altman Z-Score 計算錯誤: {zscore_results['error']}")
                    
                    st.divider()
                    
                    # 階段三：杜邦分析表格
                    st.subheader("📊 階段三：杜邦分析")
                    
                    if "error" not in dupont_results:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 年度杜邦分析表格
                            dupont_yearly_data = []
                            for year_result in dupont_results['yearly_analysis']:
                                dupont_yearly_data.append({
                                    "年度": year_result['date'],
                                    "淨利率": f"{year_result['net_margin']*100:.2f}%",
                                    "資產周轉率": f"{year_result['asset_turnover']:.4f}",
                                    "權益乘數": f"{year_result['equity_multiplier']:.4f}",
                                    "ROE (計算)": f"{year_result['roe_calculated']*100:.2f}%",
                                    "ROE (直接)": f"{year_result['roe_direct']*100:.2f}%"
                                })
                            st.dataframe(pd.DataFrame(dupont_yearly_data), use_container_width=True, hide_index=True)
                            
                            # 趨勢分析表格
                            if "error" not in dupont_results['trend_analysis']:
                                st.markdown("**趨勢變化分析**")
                                trend_data = dupont_results['trend_analysis']
                                dupont_trend_data = [
                                    {"因子": "淨利率變化", "數值": f"{trend_data['net_margin_change']*100:+.2f}%"},
                                    {"因子": "資產周轉率變化", "數值": f"{trend_data['asset_turnover_change']:+.4f}"},
                                    {"因子": "權益乘數變化", "數值": f"{trend_data['equity_multiplier_change']:+.4f}"},
                                    {"因子": "ROE變化", "數值": f"{trend_data['roe_change']*100:+.2f}%"}
                                ]
                                st.dataframe(pd.DataFrame(dupont_trend_data), use_container_width=True, hide_index=True)
                        
                        with col2:
                            # 當前ROE顯示
                            if dupont_results['yearly_analysis']:
                                current_roe = dupont_results['yearly_analysis'][0]['roe_direct'] * 100
                                st.metric("當前ROE", f"{current_roe:.2f}%")
                    else:
                        st.error(f"杜邦分析計算錯誤: {dupont_results['error']}")
                    
                    st.divider()
                    
                    # 階段四：現金流分析表格
                    st.subheader("📊 階段四：現金流分析")
                    
                    if "error" not in cashflow_results:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 現金流關鍵指標表格
                            cashflow_key_data = [
                                {"指標": "營運現金流品質比率", "數值": f"{cashflow_results['cf_quality_ratio']:.4f}"},
                                {"指標": "自由現金流", "數值": f"{cashflow_results['free_cash_flow']:,.0f}"},
                                {"指標": "現金流品質評估", "數值": cashflow_results['quality_assessment']}
                            ]
                            st.dataframe(pd.DataFrame(cashflow_key_data), use_container_width=True, hide_index=True)
                            
                            # 現金流結構分析表格
                            st.markdown("**現金流結構分析**")
                            structure = cashflow_results['structure_analysis']
                            cashflow_structure_data = [
                                {"現金流類型": "營運現金流", "金額": f"{cashflow_results['detailed_flows']['operating_cf']:,.0f}"},
                                {"現金流類型": "投資現金流", "金額": f"{cashflow_results['detailed_flows']['investing_cf']:,.0f}"},
                                {"現金流類型": "融資現金流", "金額": f"{cashflow_results['detailed_flows']['financing_cf']:,.0f}"}
                            ]
                            st.dataframe(pd.DataFrame(cashflow_structure_data), use_container_width=True, hide_index=True)
                            
                            # 詳細現金流數據
                            st.markdown("**詳細現金流數據**")
                            flows = cashflow_results['detailed_flows']
                            cashflow_detail_data = [
                                {"項目": "營運現金流", "金額": f"{flows['operating_cf']:,.0f}"},
                                {"項目": "投資現金流", "金額": f"{flows['investing_cf']:,.0f}"},
                                {"項目": "融資現金流", "金額": f"{flows['financing_cf']:,.0f}"},
                                {"項目": "淨利潤", "金額": f"{flows['net_income']:,.0f}"},
                                {"項目": "資本支出", "金額": f"{flows['capex']:,.0f}"},
                                {"項目": "現金流總計", "金額": f"{flows['total_cf']:,.0f}"}
                            ]
                            st.dataframe(pd.DataFrame(cashflow_detail_data), use_container_width=True, hide_index=True)
                        
                        with col2:
                            # 現金流品質指標顯示
                            cf_quality = cashflow_results['cf_quality_ratio']
                            quality_assessment = cashflow_results['quality_assessment']
                            
                            # 根據品質評估設定顏色
                            if quality_assessment == "優秀":
                                quality_emoji = "🟢"
                            elif quality_assessment == "良好":
                                quality_emoji = "🟡"
                            else:
                                quality_emoji = "🔴"
                            
                            st.metric("現金流品質比率", f"{cf_quality:.2f}", 
                                     delta=f"{quality_emoji} {quality_assessment}")
                    else:
                        st.error(f"現金流分析計算錯誤: {cashflow_results['error']}")
                
                except Exception as e:
                    st.error(f"四階段財報分析時發生錯誤：{str(e)}")
            
            with tab5:
                st.subheader("AI財務分析")
                
                if openai_api_key:
                    try:
                        # 準備完整的分析數據（包含計算結果和原始數據）
                        comprehensive_data = prepare_comprehensive_analysis_data(fmp_data, ticker)
                        
                        # 使用OpenAI分析數據
                        with st.spinner("正在使用AI進行四階段財務分析..."):
                            analysis = analyze_with_openai(comprehensive_data, openai_api_key, ticker)
                            st.markdown(analysis)
                    except Exception as e:
                        st.error(f"AI分析時發生錯誤：{str(e)}")
                else:
                    st.warning("請在側邊欄輸入OpenAI API金鑰以獲取AI分析")
        
        except Exception as e:
            st.error(f"發生錯誤：{str(e)}")
    
    st.sidebar.markdown("""
### 📢 免責聲明
本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。
請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
""")

if __name__ == "__main__":
    main()
