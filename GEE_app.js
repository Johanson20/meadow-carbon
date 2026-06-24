// List of attributes and years, prepared as objects from assets
var assets = {
              "1986": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1986, "ANPP Standard Error (g C/m2)": ANPP_SD_1986, "Belowground Net Primary Productivity (g C/m2)": BNPP_1986,
              "BNPP Standard Error (g C/m2)": BNPP_1986_C, "Net Ecosystem Productivity (g C/m2)": NEP_1986, "NEP Standard Error (g C/m2)": NEP_1986_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1986, "Rh Standard Error (g C/m2)": Rh_SD_1986, "Soil Carbon Concentration (%)": PercentC_1990, "Soil Carbon Stock (kg C/m2)": SoilC_1990},
              
              "1987": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1987, "ANPP Standard Error (g C/m2)": ANPP_SD_1987, "Belowground Net Primary Productivity (g C/m2)": BNPP_1987,
              "BNPP Standard Error (g C/m2)": BNPP_1987_C, "Net Ecosystem Productivity (g C/m2)": NEP_1987, "NEP Standard Error (g C/m2)": NEP_1988_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1987, "Rh Standard Error (g C/m2)": Rh_SD_1987, "Soil Carbon Concentration (%)": PercentC_1990, "Soil Carbon Stock (kg C/m2)": SoilC_1990},
              
              "1988": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1988, "ANPP Standard Error (g C/m2)": ANPP_SD_1988, "Belowground Net Primary Productivity (g C/m2)": BNPP_1988,
              "BNPP Standard Error (g C/m2)": BNPP_1988_C, "Net Ecosystem Productivity (g C/m2)": NEP_1988, "NEP Standard Error (g C/m2)": NEP_1989_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1988, "Rh Standard Error (g C/m2)": Rh_SD_1988, "Soil Carbon Concentration (%)": PercentC_1990, "Soil Carbon Stock (kg C/m2)": SoilC_1990},
              
              "1989": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1989, "ANPP Standard Error (g C/m2)": ANPP_SD_1989, "Belowground Net Primary Productivity (g C/m2)": BNPP_1989,
              "BNPP Standard Error (g C/m2)": BNPP_1989_C, "Net Ecosystem Productivity (g C/m2)": NEP_1989, "NEP Standard Error (g C/m2)": NEP_1990_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1989, "Rh Standard Error (g C/m2)": Rh_SD_1989, "Soil Carbon Concentration (%)": PercentC_1990, "Soil Carbon Stock (kg C/m2)": SoilC_1990},
              
              "1990": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1990, "ANPP Standard Error (g C/m2)": ANPP_SD_1990, "Belowground Net Primary Productivity (g C/m2)": BNPP_1990,
              "BNPP Standard Error (g C/m2)": BNPP_1990_C, "Net Ecosystem Productivity (g C/m2)": NEP_1990, "NEP Standard Error (g C/m2)": NEP_1991_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1990, "Rh Standard Error (g C/m2)": Rh_SD_1990, "Soil Carbon Concentration (%)": PercentC_1990, "Soil Carbon Stock (kg C/m2)": SoilC_1990},
              
              "1991": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1991, "ANPP Standard Error (g C/m2)": ANPP_SD_1991, "Belowground Net Primary Productivity (g C/m2)": BNPP_1991,
              "BNPP Standard Error (g C/m2)": BNPP_1991_C, "Net Ecosystem Productivity (g C/m2)": NEP_1991, "NEP Standard Error (g C/m2)": NEP_1991_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1991, "Rh Standard Error (g C/m2)": Rh_SD_1991, "Soil Carbon Concentration (%)": PercentC_1995, "Soil Carbon Stock (kg C/m2)": SoilC_1995},
              
              "1992": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1992, "ANPP Standard Error (g C/m2)": ANPP_SD_1992, "Belowground Net Primary Productivity (g C/m2)": BNPP_1992,
              "BNPP Standard Error (g C/m2)": BNPP_1992_C, "Net Ecosystem Productivity (g C/m2)": NEP_1992, "NEP Standard Error (g C/m2)": NEP_1992_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1992, "Rh Standard Error (g C/m2)": Rh_SD_1992, "Soil Carbon Concentration (%)": PercentC_1995, "Soil Carbon Stock (kg C/m2)": SoilC_1995},
              
              "1993": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1993, "ANPP Standard Error (g C/m2)": ANPP_SD_1993, "Belowground Net Primary Productivity (g C/m2)": BNPP_1993,
              "BNPP Standard Error (g C/m2)": BNPP_1993_C, "Net Ecosystem Productivity (g C/m2)": NEP_1993, "NEP Standard Error (g C/m2)": NEP_1993_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1993, "Rh Standard Error (g C/m2)": Rh_SD_1993, "Soil Carbon Concentration (%)": PercentC_1995, "Soil Carbon Stock (kg C/m2)": SoilC_1995},
              
              "1994": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1994, "ANPP Standard Error (g C/m2)": ANPP_SD_1994, "Belowground Net Primary Productivity (g C/m2)": BNPP_1994,
              "BNPP Standard Error (g C/m2)": BNPP_1994_C, "Net Ecosystem Productivity (g C/m2)": NEP_1994, "NEP Standard Error (g C/m2)": NEP_1994_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1994, "Rh Standard Error (g C/m2)": Rh_SD_1994, "Soil Carbon Concentration (%)": PercentC_1995, "Soil Carbon Stock (kg C/m2)": SoilC_1995},
              
              "1995": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1995, "ANPP Standard Error (g C/m2)": ANPP_SD_1995, "Belowground Net Primary Productivity (g C/m2)": BNPP_1995,
              "BNPP Standard Error (g C/m2)": BNPP_1995_C, "Net Ecosystem Productivity (g C/m2)": NEP_1995, "NEP Standard Error (g C/m2)": NEP_1995_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1995, "Rh Standard Error (g C/m2)": Rh_SD_1995, "Soil Carbon Concentration (%)": PercentC_1995, "Soil Carbon Stock (kg C/m2)": SoilC_1995},
              
              "1996": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1996, "ANPP Standard Error (g C/m2)": ANPP_SD_1996, "Belowground Net Primary Productivity (g C/m2)": BNPP_1996,
              "BNPP Standard Error (g C/m2)": BNPP_1996_C, "Net Ecosystem Productivity (g C/m2)": NEP_1996, "NEP Standard Error (g C/m2)": NEP_1996_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1996, "Rh Standard Error (g C/m2)": Rh_SD_1996, "Soil Carbon Concentration (%)": PercentC_2000, "Soil Carbon Stock (kg C/m2)": SoilC_2000},
              
              "1997": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1997, "ANPP Standard Error (g C/m2)": ANPP_SD_1997, "Belowground Net Primary Productivity (g C/m2)": BNPP_1997,
              "BNPP Standard Error (g C/m2)": BNPP_1997_C, "Net Ecosystem Productivity (g C/m2)": NEP_1997, "NEP Standard Error (g C/m2)": NEP_1997_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1997, "Rh Standard Error (g C/m2)": Rh_SD_1997, "Soil Carbon Concentration (%)": PercentC_2000, "Soil Carbon Stock (kg C/m2)": SoilC_2000},
              
              "1998": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1998, "ANPP Standard Error (g C/m2)": ANPP_SD_1998, "Belowground Net Primary Productivity (g C/m2)": BNPP_1998,
              "BNPP Standard Error (g C/m2)": BNPP_1998_C, "Net Ecosystem Productivity (g C/m2)": NEP_1998, "NEP Standard Error (g C/m2)": NEP_1998_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1998, "Rh Standard Error (g C/m2)": Rh_SD_1998, "Soil Carbon Concentration (%)": PercentC_2000, "Soil Carbon Stock (kg C/m2)": SoilC_2000},
              
              "1999": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_1999, "ANPP Standard Error (g C/m2)": ANPP_SD_1999, "Belowground Net Primary Productivity (g C/m2)": BNPP_1999,
              "BNPP Standard Error (g C/m2)": BNPP_1999_C, "Net Ecosystem Productivity (g C/m2)": NEP_1999, "NEP Standard Error (g C/m2)": NEP_1999_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_1999, "Rh Standard Error (g C/m2)": NEP_SD_1999, "Soil Carbon Concentration (%)": PercentC_2000, "Soil Carbon Stock (kg C/m2)": SoilC_2000},
              
              "2000": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2000, "ANPP Standard Error (g C/m2)": ANPP_SD_2000, "Belowground Net Primary Productivity (g C/m2)": BNPP_2000,
              "BNPP Standard Error (g C/m2)": BNPP_2000_C, "Net Ecosystem Productivity (g C/m2)": NEP_2000, "NEP Standard Error (g C/m2)": NEP_2000_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2000, "Rh Standard Error (g C/m2)": Rh_SD_2000, "Soil Carbon Concentration (%)": PercentC_2000, "Soil Carbon Stock (kg C/m2)": SoilC_2000},
              
              "2001": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2001, "ANPP Standard Error (g C/m2)": ANPP_SD_2001, "Belowground Net Primary Productivity (g C/m2)": BNPP_2001,
              "BNPP Standard Error (g C/m2)": BNPP_2001_C, "Net Ecosystem Productivity (g C/m2)": NEP_2001, "NEP Standard Error (g C/m2)": NEP_2001_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2001, "Rh Standard Error (g C/m2)": Rh_SD_2001, "Soil Carbon Concentration (%)": PercentC_2005, "Soil Carbon Stock (kg C/m2)": SoilC_2005},
              
              "2002": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2002, "ANPP Standard Error (g C/m2)": ANPP_SD_2002, "Belowground Net Primary Productivity (g C/m2)": BNPP_2002,
              "BNPP Standard Error (g C/m2)": BNPP_2002_C, "Net Ecosystem Productivity (g C/m2)": NEP_2002, "NEP Standard Error (g C/m2)": NEP_2002_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2002, "Rh Standard Error (g C/m2)": Rh_SD_2002, "Soil Carbon Concentration (%)": PercentC_2005, "Soil Carbon Stock (kg C/m2)": SoilC_2005},
              
              "2003": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2003, "ANPP Standard Error (g C/m2)": ANPP_SD_2003, "Belowground Net Primary Productivity (g C/m2)": BNPP_2003,
              "BNPP Standard Error (g C/m2)": BNPP_2003_C, "Net Ecosystem Productivity (g C/m2)": NEP_2003, "NEP Standard Error (g C/m2)": NEP_2003_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2003, "Rh Standard Error (g C/m2)": Rh_SD_2003, "Soil Carbon Concentration (%)": PercentC_2005, "Soil Carbon Stock (kg C/m2)": SoilC_2005},
              
              "2004": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2004, "ANPP Standard Error (g C/m2)": ANPP_SD_2004, "Belowground Net Primary Productivity (g C/m2)": BNPP_2004,
              "BNPP Standard Error (g C/m2)": BNPP_2004_C, "Net Ecosystem Productivity (g C/m2)": NEP_2004, "NEP Standard Error (g C/m2)": NEP_2004_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2004, "Rh Standard Error (g C/m2)": Rh_SD_2004, "Soil Carbon Concentration (%)": PercentC_2005, "Soil Carbon Stock (kg C/m2)": SoilC_2005},
              
              "2005": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2005, "ANPP Standard Error (g C/m2)": ANPP_SD_2005, "Belowground Net Primary Productivity (g C/m2)": BNPP_2005,
              "BNPP Standard Error (g C/m2)": BNPP_2005_C, "Net Ecosystem Productivity (g C/m2)": NEP_2005, "NEP Standard Error (g C/m2)": NEP_2005_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2005, "Rh Standard Error (g C/m2)": Rh_SD_2005, "Soil Carbon Concentration (%)": PercentC_2005, "Soil Carbon Stock (kg C/m2)": SoilC_2005},
              
              "2006": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2006, "ANPP Standard Error (g C/m2)": ANPP_SD_2006, "Belowground Net Primary Productivity (g C/m2)": BNPP_2006,
              "BNPP Standard Error (g C/m2)": BNPP_2006_C, "Net Ecosystem Productivity (g C/m2)": NEP_2006, "NEP Standard Error (g C/m2)": NEP_2006_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2006, "Rh Standard Error (g C/m2)": Rh_SD_2006, "Soil Carbon Concentration (%)": PercentC_2010, "Soil Carbon Stock (kg C/m2)": SoilC_2010},
              
              "2007": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2007, "ANPP Standard Error (g C/m2)": ANPP_SD_2007, "Belowground Net Primary Productivity (g C/m2)": BNPP_2007,
              "BNPP Standard Error (g C/m2)": BNPP_2007_C, "Net Ecosystem Productivity (g C/m2)": NEP_2007, "NEP Standard Error (g C/m2)": NEP_2007_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2007, "Rh Standard Error (g C/m2)": Rh_SD_2007, "Soil Carbon Concentration (%)": PercentC_2010, "Soil Carbon Stock (kg C/m2)": SoilC_2010},
              
              "2008": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2008, "ANPP Standard Error (g C/m2)": ANPP_SD_2008, "Belowground Net Primary Productivity (g C/m2)": BNPP_2008,
              "BNPP Standard Error (g C/m2)": BNPP_2008_C, "Net Ecosystem Productivity (g C/m2)": NEP_2008, "NEP Standard Error (g C/m2)": NEP_2008_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2008, "Rh Standard Error (g C/m2)": Rh_SD_2008, "Soil Carbon Concentration (%)": PercentC_2010, "Soil Carbon Stock (kg C/m2)": SoilC_2010},
              
              "2009": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2009, "ANPP Standard Error (g C/m2)": ANPP_SD_2009, "Belowground Net Primary Productivity (g C/m2)": BNPP_2009,
              "BNPP Standard Error (g C/m2)": BNPP_2009_C, "Net Ecosystem Productivity (g C/m2)": NEP_2009, "NEP Standard Error (g C/m2)": NEP_2009_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2009, "Rh Standard Error (g C/m2)": Rh_SD_2009, "Soil Carbon Concentration (%)": PercentC_2010, "Soil Carbon Stock (kg C/m2)": SoilC_2010},
              
              "2010": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2010, "ANPP Standard Error (g C/m2)": ANPP_SD_2010, "Belowground Net Primary Productivity (g C/m2)": BNPP_2010,
              "BNPP Standard Error (g C/m2)": BNPP_2010_C, "Net Ecosystem Productivity (g C/m2)": NEP_2010, "NEP Standard Error (g C/m2)": NEP_2010_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2010, "Rh Standard Error (g C/m2)": Rh_SD_2010, "Soil Carbon Concentration (%)": PercentC_2010, "Soil Carbon Stock (kg C/m2)": SoilC_2010},
              
              "2011": {"Aboveground Net Primary Productivity (g C/m2)": AET_2011, "ANPP Standard Error (g C/m2)": ANPP_SD_2011, "Belowground Net Primary Productivity (g C/m2)": AGD_2011,
              "BNPP Standard Error (g C/m2)": BNPP_2011_C, "Net Ecosystem Productivity (g C/m2)": NEP_2011, "NEP Standard Error (g C/m2)": NEP_2011_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2011, "Rh Standard Error (g C/m2)": Rh_SD_2011, "Soil Carbon Concentration (%)": PercentC_2015, "Soil Carbon Stock (kg C/m2)": SoilC_2015},
              
              "2012": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2012, "ANPP Standard Error (g C/m2)": ANPP_SD_2012, "Belowground Net Primary Productivity (g C/m2)": BNPP_2012,
              "BNPP Standard Error (g C/m2)": BNPP_2012_C, "Net Ecosystem Productivity (g C/m2)": NEP_2012, "NEP Standard Error (g C/m2)": NEP_2012_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2012, "Rh Standard Error (g C/m2)": Rh_SD_2012, "Soil Carbon Concentration (%)": PercentC_2015, "Soil Carbon Stock (kg C/m2)": SoilC_2015},
              
              "2013": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2013, "ANPP Standard Error (g C/m2)": ANPP_SD_2013, "Belowground Net Primary Productivity (g C/m2)": BNPP_2013,
              "BNPP Standard Error (g C/m2)": BNPP_2013_C, "Net Ecosystem Productivity (g C/m2)": NEP_2013, "NEP Standard Error (g C/m2)": NEP_2013_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2013, "Rh Standard Error (g C/m2)": Rh_SD_2013, "Soil Carbon Concentration (%)": PercentC_2015, "Soil Carbon Stock (kg C/m2)": SoilC_2015},
              
              "2014": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2014, "ANPP Standard Error (g C/m2)": ANPP_SD_2014, "Belowground Net Primary Productivity (g C/m2)": BNPP_2014,
              "BNPP Standard Error (g C/m2)": BNPP_2014_C, "Net Ecosystem Productivity (g C/m2)": NEP_2014, "NEP Standard Error (g C/m2)": NEP_2014_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2014, "Rh Standard Error (g C/m2)": Rh_SD_2014, "Soil Carbon Concentration (%)": PercentC_2015, "Soil Carbon Stock (kg C/m2)": SoilC_2015},
              
              "2015": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2015, "ANPP Standard Error (g C/m2)": ANPP_SD_2015, "Belowground Net Primary Productivity (g C/m2)": BNPP_2015,
              "BNPP Standard Error (g C/m2)": BNPP_2015_C, "Net Ecosystem Productivity (g C/m2)": NEP_2015, "NEP Standard Error (g C/m2)": NEP_2015_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2015, "Rh Standard Error (g C/m2)": Rh_SD_2015, "Soil Carbon Concentration (%)": PercentC_2015, "Soil Carbon Stock (kg C/m2)": SoilC_2015},
              
              "2016": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2016, "ANPP Standard Error (g C/m2)": ANPP_SD_2016, "Belowground Net Primary Productivity (g C/m2)": BNPP_2016,
              "BNPP Standard Error (g C/m2)": BNPP_2016_C, "Net Ecosystem Productivity (g C/m2)": NEP_2016, "NEP Standard Error (g C/m2)": NEP_2016_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2016, "Rh Standard Error (g C/m2)": Rh_SD_2016, "Soil Carbon Concentration (%)": PercentC_2020, "Soil Carbon Stock (kg C/m2)": SoilC_2020},
              
              "2017": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2017, "ANPP Standard Error (g C/m2)": ANPP_SD_2017, "Belowground Net Primary Productivity (g C/m2)": BNPP_2017,
              "BNPP Standard Error (g C/m2)": BNPP_2017_C, "Net Ecosystem Productivity (g C/m2)": NEP_2017, "NEP Standard Error (g C/m2)": NEP_2017_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2017, "Rh Standard Error (g C/m2)": Rh_SD_2017, "Soil Carbon Concentration (%)": PercentC_2020, "Soil Carbon Stock (kg C/m2)": SoilC_2020},
              
              "2018": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2018, "ANPP Standard Error (g C/m2)": ANPP_SD_2018, "Belowground Net Primary Productivity (g C/m2)": BNPP_2018,
              "BNPP Standard Error (g C/m2)": BNPP_2018_C, "Net Ecosystem Productivity (g C/m2)": NEP_2018, "NEP Standard Error (g C/m2)": NEP_2018_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2018, "Rh Standard Error (g C/m2)": Rh_SD_2018, "Soil Carbon Concentration (%)": PercentC_2020, "Soil Carbon Stock (kg C/m2)": SoilC_2020},
              
              "2019": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2019, "ANPP Standard Error (g C/m2)": ANPP_SD_2019, "Belowground Net Primary Productivity (g C/m2)": BNPP_2019,
              "BNPP Standard Error (g C/m2)": BNPP_2019_C, "Net Ecosystem Productivity (g C/m2)": NEP_2019, "NEP Standard Error (g C/m2)": NEP_2019_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2019, "Rh Standard Error (g C/m2)": Rh_SD_2019, "Soil Carbon Concentration (%)": PercentC_2020, "Soil Carbon Stock (kg C/m2)": SoilC_2020},
              
              "2020": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2020, "ANPP Standard Error (g C/m2)": ANPP_SD_2020, "Belowground Net Primary Productivity (g C/m2)": BNPP_2020,
              "BNPP Standard Error (g C/m2)": BNPP_2020_C, "Net Ecosystem Productivity (g C/m2)": NEP_2020, "NEP Standard Error (g C/m2)": NEP_2020_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2020, "Rh Standard Error (g C/m2)": Rh_SD_2020, "Soil Carbon Concentration (%)": PercentC_2020, "Soil Carbon Stock (kg C/m2)": SoilC_2020},
              
              "2021": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2021, "ANPP Standard Error (g C/m2)": ANPP_SD_2021, "Belowground Net Primary Productivity (g C/m2)": BNPP_2021,
              "BNPP Standard Error (g C/m2)": BNPP_2021_C, "Net Ecosystem Productivity (g C/m2)": NEP_2021, "NEP Standard Error (g C/m2)": NEP_2021_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2021, "Rh Standard Error (g C/m2)": Rh_SD_2021, "Soil Carbon Concentration (%)": PercentC_2024, "Soil Carbon Stock (kg C/m2)": SoilC_2024},
              
              "2022": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2022, "ANPP Standard Error (g C/m2)": ANPP_SD_2022, "Belowground Net Primary Productivity (g C/m2)": BNPP_2022,
              "BNPP Standard Error (g C/m2)": BNPP_2022_C, "Net Ecosystem Productivity (g C/m2)": NEP_2022, "NEP Standard Error (g C/m2)": NEP_2022_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2022, "Rh Standard Error (g C/m2)": Rh_SD_2022, "Soil Carbon Concentration (%)": PercentC_2024, "Soil Carbon Stock (kg C/m2)": SoilC_2024},
              
              "2023": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2023, "ANPP Standard Error (g C/m2)": ANPP_SD_2023, "Belowground Net Primary Productivity (g C/m2)": BNPP_2023,
              "BNPP Standard Error (g C/m2)": BNPP_2023_C, "Net Ecosystem Productivity (g C/m2)": NEP_2023, "NEP Standard Error (g C/m2)": NEP_2023_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2023, "Rh Standard Error (g C/m2)": Rh_SD_2023, "Soil Carbon Concentration (%)": PercentC_2024, "Soil Carbon Stock (kg C/m2)": SoilC_2024},
              
              "2024": {"Aboveground Net Primary Productivity (g C/m2)": ANPP_2024, "ANPP Standard Error (g C/m2)": ANPP_SD_2024, "Belowground Net Primary Productivity (g C/m2)": BNPP_2024,
              "BNPP Standard Error (g C/m2)": BNPP_2024_C, "Net Ecosystem Productivity (g C/m2)": NEP_2024, "NEP Standard Error (g C/m2)": NEP_2024_C, 
              "Heterotrophic Respiration (g C/m2)": Rh_2024, "Rh Standard Error (g C/m2)": Rh_SD_2024, "Soil Carbon Concentration (%)": PercentC_2024, "Soil Carbon Stock (kg C/m2)": SoilC_2024},
};

// clear any drawings: includes layers and objects
Map.drawingTools().clear();
Map.drawingTools().layers().reset();
// extract all years/attributes and create the selection panel
var years = Object.keys(assets);
var attributes = Object.keys(assets[years[0]]);
// set year to most recent year (2024)
var yearSelect = ui.Select({items: years, value: years[years.length - 1], style: {margin: '0 0 6px 0', width: '100px', height: '40px'} });
var selectPanel = ui.Panel({style: {position: 'top-left', padding: '8px'}});
selectPanel.add(ui.Label('Select Water Year/Attribute to Display', {fontWeight: 'bold'}));
selectPanel.add(yearSelect);

// initialize checkbox and add it to the selection panel for attributes
var checkboxPanel = ui.Panel();
selectPanel.add(checkboxPanel);
// add a layer for meadows
var outline = ee.Image().byte().paint({featureCollection: meadows, color: 1, width: 3});
var meadowPanel = ui.Panel({style: {position: 'middle-left', padding: '8px'}});
meadowPanel.add(ui.Label('Meadow layer', {fontWeight: 'bold'}));
// a left-panel containing meadow layer and year/attribute selection
var leftPanel = ui.Panel({layout: ui.Panel.Layout.flow('vertical'),
                          style: {position: 'top-left', padding: '8px'}});
leftPanel.add(selectPanel);
leftPanel.add(meadowPanel);
Map.add(leftPanel);

// Create polygon layer for meadows
var meadowLayer = ui.Map.Layer(meadows.style({color: 'brown', fillColor: '00000000', width: 2}),  {},
'Meadows', false);
Map.layers().add(meadowLayer);

// Create the checkbox for meadows layer
var meadowCheckbox = ui.Checkbox({label: 'Show Meadows', value: false,
  onChange: function(checked) {
    meadowLayer.setShown(checked);
    } });
meadowPanel.add(meadowCheckbox);

// initialize some variables for Map display
var legendPanel, currentImage, attri, yr, point;
var pixelInfoPanel = ui.Panel({layout: ui.Panel.Layout.flow('vertical'),
                          style: {position: 'top-center', padding: '8px'}});
var pixelInfo = ui.Label('', {whiteSpace: 'pre-wrap', position: 'top-center'});
var meadowInfo = ui.Label('', {whiteSpace: 'pre-wrap', position: 'bottom-center'});
var checkboxes;

// add time series chart where they would be displayed
var yearTicks = years.map(function(y) {
  return parseInt(y);
});
var chartPanel = ui.Panel({style: {width: '400px', position: 'bottom-right'}});
Map.add(chartPanel);

// generate 2nd and 98th percentiles of each variable per year
var pLookup = {};
Object.keys(assets[years[0]]).forEach(function(attr) {
  var attrIC = ee.ImageCollection.fromImages(
    years.map(function(y) {return ee.Image(assets[y][attr]).set('year', Number(y));
    }));
  var statsPerc = attrIC.map(function(img) {return img.set(img.reduceRegion({reducer: ee.Reducer.percentile([2, 98]), geometry: img.geometry(), scale: 30, maxPixels: 1e13}));
  });
  pLookup[attr] = {p2: statsPerc.aggregate_min('b1_p2'), p98: statsPerc.aggregate_max('b1_p98')};
});

// include variables and panels for displaying drawing capabilities and image download button
var drawingTools = Map.drawingTools();
drawingTools.setShown(true);
drawingTools.setDrawModes(['polygon']);
var downloadPanel = ui.Panel({style: {position: 'top-right', padding: '4px', backgroundColor: 'rgba(255,255,255,0.9)'}});
var downloadBtn = ui.Button({label: 'Download drawn area', onClick: handleDownload, style: {fontWeight: 'bold'}});
downloadPanel.add(downloadBtn);
// panel to include warnings about downloading drawn areas
var msgPanel = ui.Panel({layout: ui.Panel.Layout.flow('vertical')});
downloadPanel.add(msgPanel);
Map.add(downloadPanel);

// function to handle downloads of drawn areas
function handleDownload() {
  // clear previous warning messages and prompt user to draw polygon 
  msgPanel.clear();
  var layers = drawingTools.layers();
  if (layers.length() === 0) {
    msgPanel.add(ui.Label('Draw a polygon first!', {color: 'red', fontWeight: 'bold'}));
    return;
  }
  var geom = layers.get(0).getEeObject();
  // area limit (example: 100 km²): display warning if area of drawn polygon is exceeded
  geom.area().divide(1e6).evaluate(function(areaKm2) {
    if (areaKm2 > 100) {
      msgPanel.add(ui.Label('Selected area of ' + areaKm2.toFixed(3) + 'km2 is too large (limit: 100 km²)!',
                            {color: 'red', fontWeight: 'bold'}));
      return;
    }
    // build geotiff of all years over drawn geometry
    var collectionImage = ee.ImageCollection(years.map(function(y) {
        return ee.Image(assets[y][attri]).rename(y); }));
    var currentImage = collectionImage.toBands().clip(geom);
    // user-controlled download (generates a download link for boundary drawn): provide single and multi band options
    var url = currentImage.getDownloadURL({scale: 30, region: geom, crs: 'EPSG:4326', filePerBand: false, name: attri.slice(0, 18)});
    var spliturl = currentImage.getDownloadURL({scale: 30, region: geom, crs: 'EPSG:4326', name: attri.slice(0, 18)});
    var link = ui.Label({value: 'Download multi-band GeoTIFF (' + attri + ')', targetUrl: url});
    var split_link = ui.Label({value: 'Download single-band GeoTIFFs (' + attri + ')', targetUrl: spliturl});
    msgPanel.add(link);
    msgPanel.add(split_link);
  });
}

// create selection options for years and attributes
function makeAttributeCheckboxes(year) {
  checkboxPanel.clear();
  checkboxes = [];
  yr = year;
  
  attributes.forEach(function(attr) {   // loop through attributes
    var checkbox = ui.Checkbox({
      label: attr,
      value: attr === 'Net Ecosystem Productivity (g C/m2)',  // Default checked if NEP exists
      style: {fontSize: '14px', margin: '0 0 2px 0'},
      onChange: function(state) {
        if (state) {
          // Uncheck all other attributes that are not selected
          checkboxes.forEach(function(cb) {
            if (cb !== checkbox) cb.setValue(false, false);
          });
          updateMap(year, attr);
        } else {
          // clears all layers and re-adds meadowLayer
          Map.layers().reset();
          if (Map.layers().indexOf(meadowLayer) === -1) {
            Map.layers().add(meadowLayer);
          }
        }
      }
    });
    checkboxes.push(checkbox);
    checkboxPanel.add(checkbox);
  });
  // make the default checkbox selection NEP
  if (checkboxes.length > 0) {
    var defaultCb = null;
    for (var i = 0; i < checkboxes.length; i++) {
      if (checkboxes[i].getLabel() === 'Net Ecosystem Productivity (g C/m2)') {
        defaultCb = checkboxes[i];
        break;
      }
    } // make default checkbox attribute the first one if NEP is missing
    if (!defaultCb && checkboxes.length > 0) {
      defaultCb = checkboxes[0];
    } // set map to default attribute found (NEP or first one)
    if (defaultCb) {
      defaultCb.setValue(true, true);
      updateMap(year, defaultCb.getLabel());
    }
  }
}

// create legend with range of values passed in
function makeLegend(attribute, visParams) {
  // add color bar to legend
  var legend = ui.Panel({
    style: {position: 'bottom-left', padding: '8px', backgroundColor: 'white'}
  });
  legend.add(ui.Label(attribute + ' Value Range', {fontWeight: 'bold', textAlign: 'center', stretch: 'horizontal'}));
  var colorBar = ui.Thumbnail({
    image: ee.Image.pixelLonLat().select(0),
    params: {bbox: [0, 0, 1, 0.1], dimensions: '300x20', format: 'png', min: 0, max: 1, palette: visParams.palette},
    style: {stretch: 'horizontal', margin: '8px 0'}
  });
  legend.add(colorBar);
  // extract limit of legend scale (2nd and 98th percentiles)
  var maxVal = pLookup[attribute]['p98'];
  var minVal = pLookup[attribute]['p2'];
  maxVal.evaluate(function(max) {
    minVal.evaluate(function(min) {
      // include min and max labels of legend and round to the nearest 10
      max = Math.ceil(max/10)*10;
      min = Math.floor(min/10)*10;
      var middle = Math.ceil((max + min)/2);
      legend.add(ui.Panel([ui.Label(min, {margin: '4px 8px', width: '60px'}), ui.Label('', {stretch: 'horizontal'}),
        ui.Label(middle, {margin: '4px 8px', width: '60px', textAlign: 'center'}), ui.Label('', {stretch: 'horizontal'}),
        ui.Label(max, {margin: '4px 8px', width: '60px', textAlign: 'right'})], ui.Panel.Layout.flow('horizontal')));
    });
  });
  return legend;
}

// function to update entire map layer when a year/attribute is changed
function updateMap(year, attribute) {
  // clears all layers and re-adds meadowLayer
  Map.layers().reset();
  if (Map.layers().indexOf(meadowLayer) === -1) {
    Map.layers().add(meadowLayer);
  }
  if (legendPanel) Map.remove(legendPanel); // remove legend
  // update timeseries if new attribute is selected
  if (point && attri != attribute){
    generateTimeSeries(point);
  }
  // update attribute selection and extract image for year/attribute
  attri = attribute;
  var img = ee.Image(assets[year][attribute]);
  
  // Calculate min/max of attribute and round to the nearest 10
  var stats = img.reduceRegion({reducer: ee.Reducer.percentile([2, 98]), geometry: img.geometry(), scale: 30, maxPixels: 1e13});
  var minVal = ee.Number(stats.get('b1_p2')).round().divide(10).floor().multiply(10);
  var maxVal = ee.Number(stats.get('b1_p98')).round().divide(10).ceil().multiply(10);
  // Style points by selected attribute
  var visParams = {palette: ['FF0000', 'FFFF00', '00FFFF', '0000FF']};
  minVal.evaluate(function(mini) {
    maxVal.evaluate(function(maxi) {
      Map.addLayer(img, {min: mini, max: maxi, palette: visParams.palette}, attribute);
    });
  });
  // redraw legend each time a new selection is made and update current image to that of year/attribute
  legendPanel = makeLegend(attribute, visParams);
  Map.add(legendPanel);
  currentImage = img;
}

// creates and displays a time series for any clicked pixel
function generateTimeSeries(point){
  // create feature list of all years (for time series) for selected variable
  var valueList = [];
  years.forEach(function(y) {
    var img = ee.Image(assets[y][attri]);
    var val = img.reduceRegion({reducer: ee.Reducer.first(),  geometry: point,  scale: 30, bestEffort: true}).get('b1');
    valueList.push(ee.Feature(null, {year: Number(y), value: val}));
  });
  
  // make list of annual values of pixel for time series
  var fc = ee.FeatureCollection(valueList);
  var stdDev = fc.reduceColumns({reducer: ee.Reducer.stdDev(), selectors: ['value']}).get('stdDev');
  stdDev.evaluate(function(sd) {
    // calculate standard deviation of time series values; show NA if value can't be computed
    var sdText = (sd !== null) ? sd.toFixed(3) : 'NA';
    var chart = ui.Chart.feature.byFeature(fc, 'year', 'value')
      .setOptions({hAxis: {title: 'Year', format: '####', ticks: yearTicks, slantedText: true, slantedTextAngle: 90}, vAxis: {title: attri},
      title: 'Pixel time series (Std Dev = ' + sdText + ')', lineWidth: 2, pointSize: 4, format: '####'});
    chartPanel.clear();
    chartPanel.add(chart);
  });
}

// if year is changed, regenerate the checkboxes (and updatemap)
yearSelect.onChange(function(year) {
  makeAttributeCheckboxes(year);
});
// load default year (2024)
makeAttributeCheckboxes(years[years.length - 1]);

// Handles displays and what happens when a pixel is clicked on the app
Map.onClick(function(coords) {
  // clear previous displayed information of pixel/point
  if (!currentImage) return;
  // coordinates of clicked point
  point = ee.Geometry.Point(coords.lon, coords.lat);
  pixelInfoPanel.clear();
  if (meadowCheckbox.getValue()) {
    var msg = 'Meadow value at (' + coords.lon.toFixed(4) + ', ' + coords.lat.toFixed(4) + '):\n';
    // extract and display attributes of clicked point
    var feature = meadows.filterBounds(point).first();
    feature.evaluate(function(v) {
      if (meadowInfo) Map.remove(meadowInfo);
      if (v) {
        var props = v.properties;
        Object.keys(props).forEach(function(key){
          msg += key + ": " + props[key] + "\n";
        });
        pixelInfoPanel.add(meadowInfo);
      } else { msg = "No meadow here!"; }
      meadowInfo.setValue(msg);
    });
  }
  if (currentImage) {
    // extract pixel value when clicked
    var value = currentImage.sample(point, 30).first().get('b1');
    // display pixel value at the top
    value.evaluate(function(v) {
      if (pixelInfo) Map.remove(pixelInfo);
      if (v) {
        var msg = attri + ' value at (' + coords.lon.toFixed(4) + ', ' + coords.lat.toFixed(4) + ') for ' + yr + ' = ' + (v).toFixed(3);
        pixelInfoPanel.add(pixelInfo);
      } else { msg = "No data or Outside dataset boundary!"; }
      pixelInfo.setValue(msg);
    });
    if (pixelInfoPanel) Map.remove(pixelInfoPanel);
    Map.add(pixelInfoPanel);
    // generate time series for all years of a clicked pixel and display
    generateTimeSeries(point);
  }
});

// display map to appropriate zoom level (8) of entire region
Map.centerObject(currentImage, 8);