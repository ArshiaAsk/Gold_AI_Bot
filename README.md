Step 1 (Scraping) => -fetch data with web scraping "www.tgju.org" web site.
        -include 210 rows data from Gold price(IRR), USD(IRR) and Oil(Dollar).
        -Then merge each of them to a main dataset

Step 2 (Cleaning) => -Set "Date" column to datetime index and sort rows with that.
                    -convert other columns to numeric.
                    -fillna and reset index

Step 3 (Feature Engineering) => -Scaled inputs(USD and Oil) with standard scaler.
                                -create log column for variables with numpy log method
                                -drop missing values with dropna method

step 4 (modeling) => 