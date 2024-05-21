import os


def find_files_thredds(floc,ext_in):
    frf_base = 'https://chlthredds.erdc.dren.mil/thredds/catalog/frf/'
    url_in = frf_base + floc
    soup = BeautifulSoup(requests.get(url_in).text, "html.parser")
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    ids = []
    for tag in soup.find_all('dataset', id=re.compile(ext_in)):
        ids.append(tag['name'])
    return ids

def find_files_local(floc,ext_in):
    full_path = floc
    ids = []
    for file in os.listdir(full_path):
        if file.endswith(ext_in):
            ids.append(file)
    return ids
