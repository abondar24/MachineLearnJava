package org.abondar.experimental.ml4j.ad.mnist;

import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class MnistVisualiser {

    private final double imageScale;

    private final List<INDArray> digits;

    private final String title;

    private final int gridWidth;

    private final int imageSize = 28;

    public MnistVisualiser(double imageScale, List<INDArray> digits, String title) {
       this(imageScale,digits,title,5);
    }

    public MnistVisualiser(double imageScale, List<INDArray> digits, String title, int gridWidth) {
        this.imageScale = imageScale;
        this.digits = digits;
        this.title = title;
        this.gridWidth = gridWidth;
    }


    public void visualise(){
       createFrame(getPanel());
    }


    private JPanel getPanel(){
        var panel = new JPanel();
        var rows = 0;
        panel.setLayout(new GridLayout(rows,gridWidth));

        var components = getComponents();
        components.forEach(panel::add);
        return panel;
    }

    private void createFrame(JPanel panel){
        var frame = new JFrame();

        frame.setTitle(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(panel);
        frame.setVisible(true);
        frame.pack();

    }

    private List<JLabel> getComponents(){
        List<JLabel> images = new ArrayList<>();

        digits.forEach(d->{
            var image = new BufferedImage(imageSize,imageSize,BufferedImage.TYPE_BYTE_GRAY);

            for (int i=0;i<imageSize*imageSize;i++){
                image.getRaster().setSample(i % imageSize,i / imageSize,0,(int)(255 *d.getDouble(i)));
            }

            var orig = new ImageIcon(image);
            var scaledImg = orig.getImage().getScaledInstance((int)(imageScale*imageSize),
                    (int)(imageScale*imageSize),Image.SCALE_REPLICATE);
            var scaled = new ImageIcon(scaledImg);
            images.add(new JLabel(scaled));

        });

        return images;
    }
}
