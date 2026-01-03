// List of attributes and years prepared as objects from assets
var assets = {
              /*"1984": {"Active Growth Days": AGD_1984, "AET (Annual Evapotranspiration)": AET_1984, "Annual Precipitation": Precip_1984,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1984, "ANPP StdErr": ANPP_SD_1984, "BNPP (Belowground Net Primary Productivity)": BNPP_1984,
              "BNPP StdErr": BNPP_SD_1984, "Elevation (m)": Elev_1984, "NEP (Net Ecoystem Productivity)": NEP_1984, "NEP StdErr": NEP_SD_1984, "Rh (Respiration)": Rh_1984, "Rh StdErr": Rh_SD_1984},
              
              "1985": {"Active Growth Days": AGD_1985, "AET (Annual Evapotranspiration)": AET_1985, "Annual Precipitation": Precip_1985,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1985, "ANPP StdErr": ANPP_SD_1985, "BNPP (Belowground Net Primary Productivity)": BNPP_1985,
              "BNPP StdErr": BNPP_SD_1985, "Elevation (m)": Elev_1985, "NEP (Net Ecoystem Productivity)": NEP_1985, "NEP StdErr": NEP_SD_1985, "Rh (Respiration)": Rh_1985, "Rh StdErr": Rh_SD_1985},
              */
              "1986": {"Active Growth Days": AGD_1986, "AET (Annual Evapotranspiration)": AET_1986, "Annual Precipitation": Precip_1986,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1986, "ANPP StdErr": ANPP_SD_1986, "BNPP (Belowground Net Primary Productivity)": BNPP_1986,
              "BNPP StdErr": BNPP_SD_1986, "Elevation (m)": Elev_1986, "NEP (Net Ecoystem Productivity)": NEP_1986, "NEP StdErr": NEP_SD_1986, "Rh (Respiration)": Rh_1986, "Rh StdErr": Rh_SD_1986},
              
              "1987": {"Active Growth Days": AGD_1987, "AET (Annual Evapotranspiration)": AET_1987, "Annual Precipitation": Precip_1987,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1987, "ANPP StdErr": ANPP_SD_1987, "BNPP (Belowground Net Primary Productivity)": BNPP_1987,
              "BNPP StdErr": BNPP_SD_1987, "Elevation (m)": Elev_1987, "NEP (Net Ecoystem Productivity)": NEP_1987, "NEP StdErr": NEP_SD_1987, "Rh (Respiration)": Rh_1987, "Rh StdErr": Rh_SD_1987},
              
              "1988": {"Active Growth Days": AGD_1988, "AET (Annual Evapotranspiration)": AET_1988, "Annual Precipitation": Precip_1988,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1988, "ANPP StdErr": ANPP_SD_1988, "BNPP (Belowground Net Primary Productivity)": BNPP_1988,
              "BNPP StdErr": BNPP_SD_1988, "Elevation (m)": Elev_1988, "NEP (Net Ecoystem Productivity)": NEP_1988, "NEP StdErr": NEP_SD_1988, "Rh (Respiration)": Rh_1988, "Rh StdErr": Rh_SD_1988},
              
              "1989": {"Active Growth Days": AGD_1989, "AET (Annual Evapotranspiration)": AET_1989, "Annual Precipitation": Precip_1989,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1989, "ANPP StdErr": ANPP_SD_1989, "BNPP (Belowground Net Primary Productivity)": BNPP_1989,
              "BNPP StdErr": BNPP_SD_1989, "Elevation (m)": Elev_1989, "NEP (Net Ecoystem Productivity)": NEP_1989, "NEP StdErr": NEP_SD_1989, "Rh (Respiration)": Rh_1989, "Rh StdErr": Rh_SD_1989},
              
              "1990": {"Active Growth Days": AGD_1990, "AET (Annual Evapotranspiration)": AET_1990, "Annual Precipitation": Precip_1990,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1990, "ANPP StdErr": ANPP_SD_1990, "BNPP (Belowground Net Primary Productivity)": BNPP_1990,
              "BNPP StdErr": BNPP_SD_1990, "Elevation (m)": Elev_1990, "NEP (Net Ecoystem Productivity)": NEP_1990, "NEP StdErr": NEP_SD_1990, "Rh (Respiration)": Rh_1990, "Rh StdErr": Rh_SD_1990},
              
              "1991": {"Active Growth Days": AGD_1991, "AET (Annual Evapotranspiration)": AET_1991, "Annual Precipitation": Precip_1991,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1991, "ANPP StdErr": ANPP_SD_1991, "BNPP (Belowground Net Primary Productivity)": BNPP_1991,
              "BNPP StdErr": BNPP_SD_1991, "Elevation (m)": Elev_1991, "NEP (Net Ecoystem Productivity)": NEP_1991, "NEP StdErr": NEP_SD_1991, "Rh (Respiration)": Rh_1991, "Rh StdErr": Rh_SD_1991},
              
              "1992": {"Active Growth Days": AGD_1992, "AET (Annual Evapotranspiration)": AET_1992, "Annual Precipitation": Precip_1992,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1992, "ANPP StdErr": ANPP_SD_1992, "BNPP (Belowground Net Primary Productivity)": BNPP_1992,
              "BNPP StdErr": BNPP_SD_1992, "Elevation (m)": Elev_1992, "NEP (Net Ecoystem Productivity)": NEP_1992, "NEP StdErr": NEP_SD_1992, "Rh (Respiration)": Rh_1992, "Rh StdErr": Rh_SD_1992},
              
              "1993": {"Active Growth Days": AGD_1993, "AET (Annual Evapotranspiration)": AET_1993, "Annual Precipitation": Precip_1993,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1993, "ANPP StdErr": ANPP_SD_1993, "BNPP (Belowground Net Primary Productivity)": BNPP_1993,
              "BNPP StdErr": BNPP_SD_1993, "Elevation (m)": Elev_1993, "NEP (Net Ecoystem Productivity)": NEP_1993, "NEP StdErr": NEP_SD_1993, "Rh (Respiration)": Rh_1993, "Rh StdErr": Rh_SD_1993},
              
              "1994": {"Active Growth Days": AGD_1994, "AET (Annual Evapotranspiration)": AET_1994, "Annual Precipitation": Precip_1994,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1994, "ANPP StdErr": ANPP_SD_1994, "BNPP (Belowground Net Primary Productivity)": BNPP_1994,
              "BNPP StdErr": BNPP_SD_1994, "Elevation (m)": Elev_1994, "NEP (Net Ecoystem Productivity)": NEP_1994, "NEP StdErr": NEP_SD_1994, "Rh (Respiration)": Rh_1994, "Rh StdErr": Rh_SD_1994},
              
              "1995": {"Active Growth Days": AGD_1995, "AET (Annual Evapotranspiration)": AET_1995, "Annual Precipitation": Precip_1995,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1995, "ANPP StdErr": ANPP_SD_1995, "BNPP (Belowground Net Primary Productivity)": BNPP_1995,
              "BNPP StdErr": BNPP_SD_1995, "Elevation (m)": Elev_1995, "NEP (Net Ecoystem Productivity)": NEP_1995, "NEP StdErr": NEP_SD_1995, "Rh (Respiration)": Rh_1995, "Rh StdErr": Rh_SD_1995},
              
              "1996": {"Active Growth Days": AGD_1996, "AET (Annual Evapotranspiration)": AET_1996, "Annual Precipitation": Precip_1996,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1996, "ANPP StdErr": ANPP_SD_1996, "BNPP (Belowground Net Primary Productivity)": BNPP_1996,
              "BNPP StdErr": BNPP_SD_1996, "Elevation (m)": Elev_1996, "NEP (Net Ecoystem Productivity)": NEP_1996, "NEP StdErr": NEP_SD_1996, "Rh (Respiration)": Rh_1996, "Rh StdErr": Rh_SD_1996},
              
              "1997": {"Active Growth Days": AGD_1997, "AET (Annual Evapotranspiration)": AET_1997, "Annual Precipitation": Precip_1997,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1997, "ANPP StdErr": ANPP_SD_1997, "BNPP (Belowground Net Primary Productivity)": BNPP_1997,
              "BNPP StdErr": BNPP_SD_1997, "Elevation (m)": Elev_1997, "NEP (Net Ecoystem Productivity)": NEP_1997, "NEP StdErr": NEP_SD_1997, "Rh (Respiration)": Rh_1997, "Rh StdErr": Rh_SD_1997},
              
              "1998": {"Active Growth Days": AGD_1998, "AET (Annual Evapotranspiration)": AET_1998, "Annual Precipitation": Precip_1998,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1998, "ANPP StdErr": ANPP_SD_1998, "BNPP (Belowground Net Primary Productivity)": BNPP_1998,
              "BNPP StdErr": BNPP_SD_1998, "Elevation (m)": Elev_1998, "NEP (Net Ecoystem Productivity)": NEP_1998, "NEP StdErr": NEP_SD_1998, "Rh (Respiration)": Rh_1998, "Rh StdErr": Rh_SD_1998},
              
              "1999": {"Active Growth Days": AGD_1999, "AET (Annual Evapotranspiration)": AET_1999, "Annual Precipitation": Precip_1999,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_1999, "ANPP StdErr": ANPP_SD_1999, "BNPP (Belowground Net Primary Productivity)": BNPP_1999,
              "BNPP StdErr": BNPP_SD_1999, "Elevation (m)": Elev_1999, "NEP (Net Ecoystem Productivity)": NEP_1999, "NEP StdErr": NEP_SD_1999, "Rh (Respiration)": Rh_1999, "Rh StdErr": NEP_SD_1999},
              
              "2000": {"Active Growth Days": AGD_2000, "AET (Annual Evapotranspiration)": AET_2000, "Annual Precipitation": Precip_2000,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2000, "ANPP StdErr": ANPP_SD_2000, "BNPP (Belowground Net Primary Productivity)": BNPP_2000,
              "BNPP StdErr": BNPP_SD_2000, "Elevation (m)": Elev_2000, "NEP (Net Ecoystem Productivity)": NEP_2000, "NEP StdErr": NEP_SD_2000, "Rh (Respiration)": Rh_2000, "Rh StdErr": Rh_SD_2000},
              
              "2001": {"Active Growth Days": AGD_2001, "AET (Annual Evapotranspiration)": AET_2001, "Annual Precipitation": Precip_2001,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2001, "ANPP StdErr": ANPP_SD_2001, "BNPP (Belowground Net Primary Productivity)": BNPP_2001,
              "BNPP StdErr": BNPP_SD_2001, "Elevation (m)": Elev_2001, "NEP (Net Ecoystem Productivity)": NEP_2001, "NEP StdErr": NEP_SD_2001, "Rh (Respiration)": Rh_2001, "Rh StdErr": Rh_SD_2001},
              
              "2002": {"Active Growth Days": AGD_2002, "AET (Annual Evapotranspiration)": AET_2002, "Annual Precipitation": Precip_2002,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2002, "ANPP StdErr": ANPP_SD_2002, "BNPP (Belowground Net Primary Productivity)": BNPP_2002,
              "BNPP StdErr": BNPP_SD_2002, "Elevation (m)": Elev_2002, "NEP (Net Ecoystem Productivity)": NEP_2002, "NEP StdErr": NEP_SD_2002, "Rh (Respiration)": Rh_2002, "Rh StdErr": Rh_SD_2002},
              
              "2003": {"Active Growth Days": AGD_2003, "AET (Annual Evapotranspiration)": AET_2003, "Annual Precipitation": Precip_2003,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2003, "ANPP StdErr": ANPP_SD_2003, "BNPP (Belowground Net Primary Productivity)": BNPP_2003,
              "BNPP StdErr": BNPP_SD_2003, "Elevation (m)": Elev_2003, "NEP (Net Ecoystem Productivity)": NEP_2003, "NEP StdErr": NEP_SD_2003, "Rh (Respiration)": Rh_2003, "Rh StdErr": Rh_SD_2003},
              
              "2004": {"Active Growth Days": AGD_2004, "AET (Annual Evapotranspiration)": AET_2004, "Annual Precipitation": Precip_2004,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2004, "ANPP StdErr": ANPP_SD_2004, "BNPP (Belowground Net Primary Productivity)": BNPP_2004,
              "BNPP StdErr": BNPP_SD_2004, "Elevation (m)": Elev_2004, "NEP (Net Ecoystem Productivity)": NEP_2004, "NEP StdErr": NEP_SD_2004, "Rh (Respiration)": Rh_2004, "Rh StdErr": Rh_SD_2004},
              
              "2005": {"Active Growth Days": AGD_2005, "AET (Annual Evapotranspiration)": AET_2005, "Annual Precipitation": Precip_2005,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2005, "ANPP StdErr": ANPP_SD_2005, "BNPP (Belowground Net Primary Productivity)": BNPP_2005,
              "BNPP StdErr": BNPP_SD_2005, "Elevation (m)": Elev_2005, "NEP (Net Ecoystem Productivity)": NEP_2005, "NEP StdErr": NEP_SD_2005, "Rh (Respiration)": Rh_2005, "Rh StdErr": Rh_SD_2005},
              
              "2006": {"Active Growth Days": AGD_2006, "AET (Annual Evapotranspiration)": AET_2006, "Annual Precipitation": Precip_2006,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2006, "ANPP StdErr": ANPP_SD_2006, "BNPP (Belowground Net Primary Productivity)": BNPP_2006,
              "BNPP StdErr": BNPP_SD_2006, "Elevation (m)": Elev_2006, "NEP (Net Ecoystem Productivity)": NEP_2006, "NEP StdErr": NEP_SD_2006, "Rh (Respiration)": Rh_2006, "Rh StdErr": Rh_SD_2006},
              
              "2007": {"Active Growth Days": AGD_2007, "AET (Annual Evapotranspiration)": AET_2007, "Annual Precipitation": Precip_2007,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2007, "ANPP StdErr": ANPP_SD_2007, "BNPP (Belowground Net Primary Productivity)": BNPP_2007,
              "BNPP StdErr": BNPP_SD_2007, "Elevation (m)": Elev_2007, "NEP (Net Ecoystem Productivity)": NEP_2007, "NEP StdErr": NEP_SD_2007, "Rh (Respiration)": Rh_2007, "Rh StdErr": Rh_SD_2007},
              
              "2008": {"Active Growth Days": AGD_2008, "AET (Annual Evapotranspiration)": AET_2008, "Annual Precipitation": Precip_2008,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2008, "ANPP StdErr": ANPP_SD_2008, "BNPP (Belowground Net Primary Productivity)": BNPP_2008,
              "BNPP StdErr": BNPP_SD_2008, "Elevation (m)": Elev_2008, "NEP (Net Ecoystem Productivity)": NEP_2008, "NEP StdErr": NEP_SD_2008, "Rh (Respiration)": Rh_2008, "Rh StdErr": Rh_SD_2008},
              
              "2009": {"Active Growth Days": AGD_2009, "AET (Annual Evapotranspiration)": AET_2009, "Annual Precipitation": Precip_2009,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2009, "ANPP StdErr": ANPP_SD_2009, "BNPP (Belowground Net Primary Productivity)": BNPP_2009,
              "BNPP StdErr": BNPP_SD_2009, "Elevation (m)": Elev_2009, "NEP (Net Ecoystem Productivity)": NEP_2009, "NEP StdErr": NEP_SD_2009, "Rh (Respiration)": Rh_2009, "Rh StdErr": Rh_SD_2009},
              
              "2010": {"Active Growth Days": AGD_2010, "AET (Annual Evapotranspiration)": AET_2010, "Annual Precipitation": Precip_2010,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2010, "ANPP StdErr": ANPP_SD_2010, "BNPP (Belowground Net Primary Productivity)": BNPP_2010,
              "BNPP StdErr": BNPP_SD_2010, "Elevation (m)": Elev_2010, "NEP (Net Ecoystem Productivity)": NEP_2010, "NEP StdErr": NEP_SD_2010, "Rh (Respiration)": Rh_2010, "Rh StdErr": Rh_SD_2010},
              
              "2011": {"Active Growth Days": AGD_2011, "AET (Annual Evapotranspiration)": AET_2011, "Annual Precipitation": Precip_2011,
              "ANPP (Aboveground Net Primary Productivity)": AET_2011, "ANPP StdErr": ANPP_SD_2011, "BNPP (Belowground Net Primary Productivity)": AGD_2011,
              "BNPP StdErr": BNPP_SD_2011, "Elevation (m)": Elev_2011, "NEP (Net Ecoystem Productivity)": NEP_2011, "NEP StdErr": NEP_SD_2011, "Rh (Respiration)": Rh_2011, "Rh StdErr": Rh_SD_2011},
              
              "2012": {"Active Growth Days": AGD_2012, "AET (Annual Evapotranspiration)": AET_2012, "Annual Precipitation": Precip_2012,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2012, "ANPP StdErr": ANPP_SD_2012, "BNPP (Belowground Net Primary Productivity)": BNPP_2012,
              "BNPP StdErr": BNPP_SD_2012, "Elevation (m)": Elev_2012, "NEP (Net Ecoystem Productivity)": NEP_2012, "NEP StdErr": NEP_SD_2012, "Rh (Respiration)": Rh_2012, "Rh StdErr": Rh_SD_2012},
              
              "2013": {"Active Growth Days": AGD_2013, "AET (Annual Evapotranspiration)": AET_2013, "Annual Precipitation": Precip_2013,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2013, "ANPP StdErr": ANPP_SD_2013, "BNPP (Belowground Net Primary Productivity)": BNPP_2013,
              "BNPP StdErr": BNPP_SD_2013, "Elevation (m)": Elev_2013, "NEP (Net Ecoystem Productivity)": NEP_2013, "NEP StdErr": NEP_SD_2013, "Rh (Respiration)": Rh_2013, "Rh StdErr": Rh_SD_2013},
              
              "2014": {"Active Growth Days": AGD_2014, "AET (Annual Evapotranspiration)": AET_2014, "Annual Precipitation": Precip_2014,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2014, "ANPP StdErr": ANPP_SD_2014, "BNPP (Belowground Net Primary Productivity)": BNPP_2014,
              "BNPP StdErr": BNPP_SD_2014, "Elevation (m)": Elev_2014, "NEP (Net Ecoystem Productivity)": NEP_2014, "NEP StdErr": NEP_SD_2014, "Rh (Respiration)": Rh_2014, "Rh StdErr": Rh_SD_2014},
              
              "2015": {"Active Growth Days": AGD_2015, "AET (Annual Evapotranspiration)": AET_2015, "Annual Precipitation": Precip_2015,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2015, "ANPP StdErr": ANPP_SD_2015, "BNPP (Belowground Net Primary Productivity)": BNPP_2015,
              "BNPP StdErr": BNPP_SD_2015, "Elevation (m)": Elev_2015, "NEP (Net Ecoystem Productivity)": NEP_2015, "NEP StdErr": NEP_SD_2015, "Rh (Respiration)": Rh_2015, "Rh StdErr": Rh_SD_2015},
              
              "2016": {"Active Growth Days": AGD_2016, "AET (Annual Evapotranspiration)": AET_2016, "Annual Precipitation": Precip_2016,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2016, "ANPP StdErr": ANPP_SD_2016, "BNPP (Belowground Net Primary Productivity)": BNPP_2016,
              "BNPP StdErr": BNPP_SD_2016, "Elevation (m)": Elev_2016, "NEP (Net Ecoystem Productivity)": NEP_2016, "NEP StdErr": NEP_SD_2016, "Rh (Respiration)": Rh_2016, "Rh StdErr": Rh_SD_2016},
              
              "2017": {"Active Growth Days": AGD_2017, "AET (Annual Evapotranspiration)": AET_2017, "Annual Precipitation": Precip_2017,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2017, "ANPP StdErr": ANPP_SD_2017, "BNPP (Belowground Net Primary Productivity)": BNPP_2017,
              "BNPP StdErr": BNPP_SD_2017, "Elevation (m)": Elev_2017, "NEP (Net Ecoystem Productivity)": NEP_2017, "NEP StdErr": NEP_SD_2017, "Rh (Respiration)": Rh_2017, "Rh StdErr": Rh_SD_2017},
              
              "2018": {"Active Growth Days": AGD_2018, "AET (Annual Evapotranspiration)": AET_2018, "Annual Precipitation": Precip_2018,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2018, "ANPP StdErr": ANPP_SD_2018, "BNPP (Belowground Net Primary Productivity)": BNPP_2018,
              "BNPP StdErr": BNPP_SD_2018, "Elevation (m)": Elev_2018, "NEP (Net Ecoystem Productivity)": NEP_2018, "NEP StdErr": NEP_SD_2018, "Rh (Respiration)": Rh_2018, "Rh StdErr": Rh_SD_2018},
              
              "2019": {"Active Growth Days": AGD_2019, "AET (Annual Evapotranspiration)": AET_2019, "Annual Precipitation": Precip_2019,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2019, "ANPP StdErr": ANPP_SD_2019, "BNPP (Belowground Net Primary Productivity)": BNPP_2019,
              "BNPP StdErr": BNPP_SD_2019, "Elevation (m)": Elev_2019, "NEP (Net Ecoystem Productivity)": NEP_2019, "NEP StdErr": NEP_SD_2019, "Rh (Respiration)": Rh_2019, "Rh StdErr": Rh_SD_2019},
              
              "2020": {"Active Growth Days": AGD_2020, "AET (Annual Evapotranspiration)": AET_2020, "Annual Precipitation": Precip_2020,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2020, "ANPP StdErr": ANPP_SD_2020, "BNPP (Belowground Net Primary Productivity)": BNPP_2020,
              "BNPP StdErr": BNPP_SD_2020, "Elevation (m)": Elev_2020, "NEP (Net Ecoystem Productivity)": NEP_2020, "NEP StdErr": NEP_SD_2020, "Rh (Respiration)": Rh_2020, "Rh StdErr": Rh_SD_2020},
              
              "2021": {"Active Growth Days": AGD_2021, "AET (Annual Evapotranspiration)": AET_2021, "Annual Precipitation": Precip_2021,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2021, "ANPP StdErr": ANPP_SD_2021, "BNPP (Belowground Net Primary Productivity)": BNPP_2021,
              "BNPP StdErr": BNPP_SD_2021, "Elevation (m)": Elev_2021, "NEP (Net Ecoystem Productivity)": NEP_2021, "NEP StdErr": NEP_SD_2021, "Rh (Respiration)": Rh_2021, "Rh StdErr": Rh_SD_2021},
              
              "2022": {"Active Growth Days": AGD_2022, "AET (Annual Evapotranspiration)": AET_2022, "Annual Precipitation": Precip_2022,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2022, "ANPP StdErr": ANPP_SD_2022, "BNPP (Belowground Net Primary Productivity)": BNPP_2022,
              "BNPP StdErr": BNPP_SD_2022, "Elevation (m)": Elev_2022, "NEP (Net Ecoystem Productivity)": NEP_2022, "NEP StdErr": NEP_SD_2022, "Rh (Respiration)": Rh_2022, "Rh StdErr": Rh_SD_2022},
              
              "2023": {"Active Growth Days": AGD_2023, "AET (Annual Evapotranspiration)": AET_2023, "Annual Precipitation": Precip_2023,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2023, "ANPP StdErr": ANPP_SD_2023, "BNPP (Belowground Net Primary Productivity)": BNPP_2023,
              "BNPP StdErr": BNPP_SD_2023, "Elevation (m)": Elev_2023, "NEP (Net Ecoystem Productivity)": NEP_2023, "NEP StdErr": NEP_SD_2023, "Rh (Respiration)": Rh_2023, "Rh StdErr": Rh_SD_2023},
              
              "2024": {"Active Growth Days": AGD_2024, "AET (Annual Evapotranspiration)": AET_2024, "Annual Precipitation": Precip_2024,
              "ANPP (Aboveground Net Primary Productivity)": ANPP_2024, "ANPP StdErr": ANPP_SD_2024, "BNPP (Belowground Net Primary Productivity)": BNPP_2024,
              "BNPP StdErr": BNPP_SD_2024, "Elevation (m)": Elev_2024, "NEP (Net Ecoystem Productivity)": NEP_2024, "NEP StdErr": NEP_SD_2024, "Rh (Respiration)": Rh_2024, "Rh StdErr": Rh_SD_2024},
};

// clear any drawings
Map.drawingTools().layers().reset();
// extract years/attributes and create the selection panel
var years = Object.keys(assets);
var attributes = Object.keys(assets[years[0]]);
var yearSelect = ui.Select({items: years, value: years[0], style: {margin: '0 0 6px 0', width: '100px', height: '40px'} });
// set year to last year (2024)
yearSelect.setValue(years[years.length - 1], true);
var selectPanel = ui.Panel({style: {position: 'top-left', padding: '8px'}});
selectPanel.add(ui.Label('Select Year/Attribute to Display', {fontWeight: 'bold'}));
selectPanel.add(yearSelect);

// initialize checkbox and add it to the selection panel for attributes
var checkboxPanel = ui.Panel();
selectPanel.add(checkboxPanel);
Map.add(selectPanel);

// initialize some variables for Map display
var legendPanel, currentImage, attri, yr;
var pixelInfo = ui.Label('');
var checkboxes;

// add time series chart
var yearTicks = years.map(function(y) {
  return parseInt(y);
});
var chartPanel = ui.Panel({style: {width: '400px', position: 'bottom-right'}});
Map.add(chartPanel);

// generate 98th percentiles of each variable per year and use max of all years per variable
var p98Lookup = {};
Object.keys(assets[years[0]]).forEach(function(attr) {
  var attrIC = ee.ImageCollection.fromImages(
    years.map(function(y) {return ee.Image(assets[y][attr]).set('year', Number(y));
    }));
  var maxP98 = attrIC.map(function(img) {return img.set('p98',
  img.reduceRegion({reducer: ee.Reducer.percentile([98]), geometry: img.geometry(), scale: 30, maxPixels: 1e13}).get('b1'));
  }).aggregate_max('p98');
  p98Lookup[attr] = maxP98;
});
/* quick loading of percentile values (pre-computed in python from detailed annual CSVs)
var percentiles_98 =[2001.0, 727.1, 237.0, 2983.725146484375, 412.0129242790604, 1598.1482852643521, 1202.7558636892268,
                      1315.7023544660567, 712.6666091599797, 829.6798922771411, 1386.4061729674954, 1301.5462543077829]
attributes.forEach(function(attr, i) {
  p98Lookup[attr] = percentiles_98[i];
});
*/

// include variables and panels for displaying drawing capabilities and image download
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
  msgPanel.clear();
  var layers = drawingTools.layers();
  if (layers.length() === 0) {
    msgPanel.add(ui.Label('Draw a polygon first!', {color: 'red', fontWeight: 'bold'}));
    return;
  }
  var geom = layers.get(0).getEeObject();
  // area limit (example: 100 km²)
  geom.area().divide(1e6).evaluate(function(areaKm2) {
    if (areaKm2 > 100) {
      msgPanel.add(ui.Label('Selected area of ' + areaKm2.toFixed(3) + 'km2 is too large (limit: 100 km²)!',
                            {color: 'red', fontWeight: 'bold'}));
      return;
    }
    // build multiband single image: one band per year
    var currentImage = ee.Image(years.map(function(y) {
        return ee.Image(assets[y][attri])})
        ).clip(geom);
    // user-controlled download (generates a download link for boundary drawn): provide single and multi band options
    var url = currentImage.getDownloadURL({scale: 30, region: geom, crs: 'EPSG:4326', filePerBand: false});
    var spliturl = currentImage.getDownloadURL({scale: 30, region: geom, crs: 'EPSG:4326'});
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
      value: attr === 'NEP (Net Ecoystem Productivity)',  // Default checked if NEP exists
      style: {fontSize: '14px', margin: '0 0 2px 0'},
      onChange: function(state) {
        if (state) {
          // Uncheck all others
          checkboxes.forEach(function(cb) {
            if (cb !== checkbox) cb.setValue(false, false);
          });
          updateMap(year, attr);
        } else {
          Map.layers().reset();
        }
      }
    });
    checkboxes.push(checkbox);
    checkboxPanel.add(checkbox);
  });
  // make the default checkbox selection NEP or the first attribute if NEP is missing
  if (checkboxes.length > 0) {
    var defaultCb = null;
    for (var i = 0; i < checkboxes.length; i++) {
      if (checkboxes[i].getLabel() === 'NEP (Net Ecoystem Productivity)') {
        defaultCb = checkboxes[i];
        break;
      }
    }
    if (!defaultCb && checkboxes.length > 0) {
      defaultCb = checkboxes[0];
    }
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
  // extract limit of legend scale
  var maxVal = p98Lookup[attribute];
  maxVal.evaluate(function(max) {
    // include uniform min and max labels of legend and round to the nearest 10
    max = Math.floor(max/10)*10;
    var min = -max;
    legend.add(ui.Panel([ui.Label(min, {margin: '4px 8px', width: '60px'}), ui.Label('', {stretch: 'horizontal'}),
      ui.Label(0, {margin: '4px 8px', width: '60px', textAlign: 'center'}), ui.Label('', {stretch: 'horizontal'}),
      ui.Label(max, {margin: '4px 8px', width: '60px', textAlign: 'right'})], ui.Panel.Layout.flow('horizontal')));
  });
  return legend;
}

// function to update entire map layer when a year/attribute is changed
function updateMap(year, attribute) {
  Map.layers().reset();
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
  // remove and redraw legend each time a new selection is made
  if (legendPanel) Map.remove(legendPanel);
  legendPanel = makeLegend(attribute, visParams);
  Map.add(legendPanel);
  currentImage = img;
  attri = attribute;
}

// if year is changed, regenerate the checkboxes (and updatemap)
yearSelect.onChange(function(year) {
  makeAttributeCheckboxes(year);
});
// load default year (2024)
makeAttributeCheckboxes(years[years.length - 1]);

// Extract attribute value of particular year and generate time series of all years whena pixel is clicked
Map.onClick(function(coords) {
  if (!currentImage) return;
  // extract coordinates and pixel value when clicked
  var point = ee.Geometry.Point(coords.lon, coords.lat);
  var value = currentImage.sample(point, 30).first().get('b1');
  
  // display pixel value at the top
  value.evaluate(function(v) {
    if (pixelInfo) Map.remove(pixelInfo);
    var msg = attri + ' value at (' + coords.lon.toFixed(4) + ', ' + coords.lat.toFixed(4) + ') for ' + yr + ' = ' + (v).toFixed(3);
    pixelInfo.setValue(msg);
    Map.add(pixelInfo);
  });
  
  // create feature list of all years (for time series) for selected variable
  var featureList = [];
  years.forEach(function(y) {
    var img = ee.Image(assets[y][attri]);
    var val = img.reduceRegion({reducer: ee.Reducer.first(),  geometry: point,  scale: 30, bestEffort: true}).get('b1');
    featureList.push(ee.Feature(null, {year: Number(y), value: val}));
  });
  // generate time series for clicked pixel and display
  var fc = ee.FeatureCollection(featureList);
  var stdDev = fc.reduceColumns({reducer: ee.Reducer.stdDev(), selectors: ['value']}).get('stdDev');
  stdDev.evaluate(function(sd) {
    var sdText = (sd !== null) ? sd.toFixed(3) : 'NA';
    var chart = ui.Chart.feature.byFeature(fc, 'year', 'value')
      .setOptions({hAxis: {title: 'Year', format: '####', ticks: yearTicks, slantedText: true, slantedTextAngle: 90}, vAxis: {title: attri},
      title: 'Pixel time series (Std Dev = ' + sdText + ')', lineWidth: 2, pointSize: 4, format: '####'});
    chartPanel.clear();
    chartPanel.add(chart);
  });
});

// display map to appropriate zoom level (8) of entire region
Map.centerObject(currentImage, 8);