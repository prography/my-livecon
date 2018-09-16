from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(storage={'root_dir': 'D:\\Deep_learning\\Data\\portrait\\oil-painting\\2'},
                             feeder_threads=1,
                             parser_threads=2,
                             downloader_threads=4
                             )
crawler.crawl(keyword='oil painting picture', max_num=500)