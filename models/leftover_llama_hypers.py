
#missing
# hypers = {
#     "base": 10,
#     "prec": args.prec,
#     "time_sep": args.time_sep,
#     "bit_sep": args.bit_sep,
#     "missing_str": "NaN",
# }





# promptcast_hypers = dict(
#     base=10,
#     prec=0, 
#     signed=True, 
#     time_sep=',',
#     bit_sep='',
#     plus_sign='',
#     minus_sign='-',
#     half_bin_correction=False,
#     decimal_point=''
# )
# hypers = promptcast_hypers


# beta = 0 # args.beta
# alpha = -1 # args.alpha
# prec = 0 # args.prec

# for ds_tuple in ds_tuples:
#     print(ds_tuple)

#     dsname, train_frac = ds_tuple

#     print(f"Running on {dsname}...")

#     hypers = {
#         "base": 10,
#         "prec": prec,
#         "time_sep": args.time_sep,
#         "bit_sep": args.bit_sep,
#         "signed": True,
#     }



#monash
# hypers = {
#     "base": 10,
#     "prec": args.prec,
#     "time_sep": args.time_sep,
#     "bit_sep": args.bit_sep,
#     "signed": True,
# }


#autoformer 

# dsname, series_num = ds_tuple

# if dsname == "national_illness.csv":
#     test_length = 36

# df = pd.read_csv(
#     f"/private/home/ngruver/time-series-lm/autoformer/{dsname}.csv"
# )

# train = df.iloc[:-test_length,series_num]
# test = df.iloc[-test_length:,series_num]

# hypers = {
#     "base": 10,
#     "prec": args.prec,
#     "time_sep": args.time_sep,
#     "bit_sep": args.bit_sep,
#     "signed": True,
# }


# parser.add_argument("--alpha", type=float, default=0.99)
# parser.add_argument("--beta", type=float, default=0.3)
# parser.add_argument("--prec", type=int, default=3)