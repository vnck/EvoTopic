from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from tqdm import tqdm
import csv

def get_html(url):
    '''
        Accepts a single URL argument and makes an HTTP GET request to that URL. If nothing goes wrong and
        the content-type of the response is some kind of HTMl/XML, return the raw HTML content for the
        requested page. However, if there were problems with the request, return None.
    '''
    try:
        with closing(get(url, stream=True)) as resp:
            if quality_response(resp):
                return resp.content
            else:
                return None
    except RequestException as re:
        print(f"There was an error during requests to {url} : {str(re)}")
        return None

def quality_response(resp):
    '''
        Returns true if response seems to be HTML, false otherwise.
    '''
    content_type = resp.headers["Content-Type"].lower()
    return (resp.status_code == 200 and content_type is not None and content_type.find("html") > - 1)


def get_title_description(url):
    ''' 
        Downloads the webpage, Finds the title, description and puts it into a list
    '''
    #url = "https://github.com/facebook/react/issues/17399"
    response = get_html(url)
    title_desc_small_list = []
    if response is not None:
        soup = BeautifulSoup(response, "html.parser")
        #get title of the issue
        title = soup.title.get_text()
        #create a list of titles
        title_desc_small_list.append(title)
        bottom_description = soup.find_all('task-lists')
        description = bottom_description[0].get_text().replace("\n", " ")
        title_desc_small_list.append(description)
    return title_desc_small_list
       

def write_to_csv(big_list):
    ''' 
        Accepts a single item list as an argument, proceses through the list and writes all the products into
        a single CSV data file.
    '''
    headers = ["title", "description"]
    filename = "github_issues_test.csv"
    try:
        with open(filename, "w") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(headers)
            for row in big_list:
                writer.writerow(row)    
            csvFile.close()
    except:
        print("There was an error writing to the CSV data file.")

def get_all_issues():
    
    all_links = []
    #the range here changes depending on the number of pages in the github issue page
    #iterates through all the pages in the facebook react issues.
    for i in range(1,313):
    	#you can just change this url to any other url you want to web scrape from.
        url = "https://github.com/facebook/react/issues?page="+str(i)+"&q=is%3Aissue+is%3Aclosed"
        print(url)
        response = get_html(url)

        #list to store links
        if response is not None:
            soup = BeautifulSoup(response, "html.parser") #Parse html file
            #accessing the data that we interested in
            links = soup.find_all('a', attrs={'data-hovercard-type':'issue'})
            for link in links:
                all_links.append(link.get('href'))

    big_data_list = []
    for link in tqdm(all_links):
        print(link)
        one_issue_url = "http://github.com"+str(link)
        #print("HERES THE URL:", one_issue_url)
        title_desc = get_title_description(one_issue_url)
        big_data_list.append(title_desc)

    ##WRITE TO CSV HERE WITH BIG LIST.
    print("writing")
    write_to_csv(big_data_list)

if __name__ == "__main__":
    get_all_issues()

    

