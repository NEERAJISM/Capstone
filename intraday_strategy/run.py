from common.data_loader import StockDataLoader 


loader = StockDataLoader(
    base_dir="downloaded_files",
    start="2021-08-01 09:15:00",
    end = "2021-10-31 15:30:00",
    tickers = [],
    impute= True
 
)
print(loader.get_data_for_tickers())
   
     