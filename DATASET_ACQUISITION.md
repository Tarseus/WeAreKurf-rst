# CelebDF-v2 / DFGC-21 Data Acquisition

The full datasets are not direct public downloads. The official pages require
an access request, then send download links after approval.

## Official Access Pages

- CelebDF-v2: https://github.com/yuezunli/celeb-deepfakeforensics
- DFGC-21: https://github.com/bomb2peng/DFGC_starterkit/tree/master/DFGC-21%20dataset

## Once Links Are Approved

Put one authorized direct URL per line in:

```text
datasets/authorized_urls.txt
```

Then download locally:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_authorized_dataset.ps1 `
  -UrlFile datasets\authorized_urls.txt `
  -OutputDir datasets\downloads
```

If the dataset is already downloaded locally, skip the download and sync it to
g51:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\sync_dataset_to_g51.ps1 `
  -LocalPath E:\CAS\3\安全\大作业\datasets\Celeb-DF-v2 `
  -RemotePath /data1/gushengda/deepfake_detection_dfgc/datasets `
  -RemoteName Celeb-DF-v2
```

The remote has enough capacity under `/data1`; the last check showed about
4.7 TB free.

## Why This Is Not Automatic Yet

CelebDF-v2 and DFGC-21 are released behind application forms. I can download and
sync approved links, but I should not scrape or bypass the access process.
