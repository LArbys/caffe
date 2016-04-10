from ROOT import larcv

iom = larcv.IOManager(larcv.IOManager.kWRITE)
iom.set_out_file("butthole.root")
iom.set_verbosity(0)
iom.initialize()


for i in xrange(10):

    ev_image = iom.get_data(larcv.kProductImage2D,"event_image")
    ev_roi   = iom.get_data(larcv.kProductROI,"event_roi")
    
    for ix  in xrange(3):
        img = larcv.Image2D(10,10)
        roi = larcv.ROI(larcv.kROIPizero,
                        larcv.kShapeTrack)
        ev_roi.Append( roi )

        for ii in xrange(10):
            for jj in xrange(10):
                # fill row ii along the columns
                img.set_pixel(jj, ii, ( ( i * 3 + ix ) * 10 + ii ) * 10 + jj )

                print ( ( i * 3 + ix ) * 10 + ii ) * 10 + jj
                
        ev_image.Append( img )
    
    iom.set_id(1,0,i)

    iom.save_entry()    


iom.finalize()
