from ROOT import larcv

iom = larcv.IOManager(larcv.IOManager.kWRITE)

iom.set_verbosity(0)

iom.set_out_file("butthole.root")

iom.initialize()


evi = iom.get_data(0,"event_image")

im1 = larcv.Image2D(10,10)
im2 = larcv.Image2D(10,10)
im3 = larcv.Image2D(10,10)

evi.Append(im1)
evi.Append(im2)
evi.Append(im3)

iom.set_id(1,0,0);
iom.save_entry()

evi = iom.get_data(0,"event_image")

im1 = larcv.Image2D(20,20)
im2 = larcv.Image2D(20,20)
im3 = larcv.Image2D(20,20)

evi.Append(im1)
evi.Append(im2)
evi.Append(im3)

iom.set_id(2,0,0);
iom.save_entry()


iom.finalize()
