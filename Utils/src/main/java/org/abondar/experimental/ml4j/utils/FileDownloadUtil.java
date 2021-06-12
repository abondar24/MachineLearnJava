package org.abondar.experimental.ml4j.utils;

import java.io.BufferedInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;

public class FileDownloadUtil {

    public static void downloadArchive(String urlPath, String archivePAth) throws IOException {
        var url = new URL(urlPath);
        var inputStream = new BufferedInputStream(url.openStream());
        var fos = new FileOutputStream(archivePAth);

        byte[] dataBuffer = new byte[1024];
        int bytesRead;
        while ((bytesRead = inputStream.read(dataBuffer, 0, 1024)) != -1) {
            fos.write(dataBuffer, 0, bytesRead);
        }

    }


    private FileDownloadUtil(){}
}
