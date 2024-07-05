from gdpr_rag.documents.gdpr import GDPR
from gdpr_rag.documents.article_30_5 import Article_30_5
from gdpr_rag.documents.article_47_bcr import Article_47_BCR
from gdpr_rag.documents.decision_making import DecisionMaking
from gdpr_rag.documents.dpia import DPIA
from gdpr_rag.documents.dpo import DPO
from gdpr_rag.documents.article_49_intl_transfer import Article_49_Intl_Transfer
from gdpr_rag.documents.lead_sa import Lead_SA
from gdpr_rag.documents.data_breach import DataBreach
from gdpr_rag.documents.data_portability import DataPortability
from gdpr_rag.documents.transparency import Transparency
from gdpr_rag.documents.codes import Codes
from gdpr_rag.documents.online_services import OnlineServices
from gdpr_rag.documents.territorial_scope import TerritorialScope
from gdpr_rag.documents.video import Video
from gdpr_rag.documents.covid_health import CovidHealth
from gdpr_rag.documents.covid_location import CovidLocation
from gdpr_rag.documents.consent import Consent
from gdpr_rag.documents.forgotten import Forgotten
from gdpr_rag.documents.protection import Protection



from regulations_rag.corpus import Corpus , create_document_dictionary_from_folder


class GDPRCorpus(Corpus):
    def __init__(self, folder):
        document_dictionary = create_document_dictionary_from_folder(folder, globals())
        super().__init__(document_dictionary)

    def get_primary_document(self):
        return "GDPR"


