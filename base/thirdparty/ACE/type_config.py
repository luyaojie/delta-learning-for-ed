#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/2/26
type_entity = '''
TYPE="PER" SUBTYPE="Individual"
TYPE="PER" SUBTYPE="Group"
TYPE="PER" SUBTYPE="Indeterminate"

TYPE="ORG" SUBTYPE="Government"
TYPE="ORG" SUBTYPE="Non-Governmental"
TYPE="ORG" SUBTYPE="Commercial"
TYPE="ORG" SUBTYPE="Educational"
TYPE="ORG" SUBTYPE="Media"
TYPE="ORG" SUBTYPE="Religious"
TYPE="ORG" SUBTYPE="Sports"
TYPE="ORG" SUBTYPE="Medical-Science"
TYPE="ORG" SUBTYPE="Entertainment"

TYPE="LOC" SUBTYPE="Address"
TYPE="LOC" SUBTYPE="Boundary"
TYPE="LOC" SUBTYPE="Water-Body"
TYPE="LOC" SUBTYPE="Celestial"
TYPE="LOC" SUBTYPE="Land-Region-Natural"
TYPE="LOC" SUBTYPE="Region-General"
TYPE="LOC" SUBTYPE="Region-International"

TYPE="GPE" SUBTYPE="Continent"
TYPE="GPE" SUBTYPE="Nation"
TYPE="GPE" SUBTYPE="State-or-Province"
TYPE="GPE" SUBTYPE="County-or-District"
TYPE="GPE" SUBTYPE="Population-Center"
TYPE="GPE" SUBTYPE="GPE-Cluster"
TYPE="GPE" SUBTYPE="Special"

TYPE="FAC" SUBTYPE="Building-Grounds"
TYPE="FAC" SUBTYPE="Subarea-Facility"
TYPE="FAC" SUBTYPE="Path"
TYPE="FAC" SUBTYPE="Airport"
TYPE="FAC" SUBTYPE="Plant"

TYPE="VEH" SUBTYPE="Land"
TYPE="VEH" SUBTYPE="Air"
TYPE="VEH" SUBTYPE="Water"
TYPE="VEH" SUBTYPE="Subarea-Vehicle"
TYPE="VEH" SUBTYPE="Underspecified"

TYPE="WEA" SUBTYPE="Blunt"
TYPE="WEA" SUBTYPE="Exploding"
TYPE="WEA" SUBTYPE="Sharp"
TYPE="WEA" SUBTYPE="Chemical"
TYPE="WEA" SUBTYPE="Biological"
TYPE="WEA" SUBTYPE="Shooting"
TYPE="WEA" SUBTYPE="Projectile"
TYPE="WEA" SUBTYPE="Nuclear"
TYPE="WEA" SUBTYPE="Underspecified"
'''

type_value = """
TYPE="Numeric" SUBTYPE="Money"
TYPE="Numeric" SUBTYPE="Percent"
TYPE="Contact-Info" SUBTYPE="Phone-Number"
TYPE="Contact-Info" SUBTYPE="E-Mail"
TYPE="Contact-Info" SUBTYPE="URL"

TYPE="Crime"
TYPE="Job-Title"
TYPE="Sentence"
"""

type_relation = """
TYPE="PHYS" SUBTYPE="Located"
TYPE="PHYS" SUBTYPE="Near"

TYPE="PART-WHOLE" SUBTYPE="Geographical"
TYPE="PART-WHOLE" SUBTYPE="Subsidiary"
TYPE="PART-WHOLE" SUBTYPE="Artifact"

TYPE="PER-SOC" SUBTYPE="Business"
TYPE="PER-SOC" SUBTYPE="Family"
TYPE="PER-SOC" SUBTYPE="Lasting-Personal"

TYPE="ORG-AFF" SUBTYPE="Employment"
TYPE="ORG-AFF" SUBTYPE="Ownership"
TYPE="ORG-AFF" SUBTYPE="Founder"
TYPE="ORG-AFF" SUBTYPE="Student-Alum"
TYPE="ORG-AFF" SUBTYPE="Sports-Affiliation"
TYPE="ORG-AFF" SUBTYPE="Investor-Shareholder"
TYPE="ORG-AFF" SUBTYPE="Membership"

TYPE="ART" SUBTYPE="User-Owner-Inventor-Manufacturer"

TYPE="GEN-AFF" SUBTYPE="Citizen-Resident-Religion-Ethnicity"
TYPE="GEN-AFF" SUBTYPE="Org-Location"

TYPE="METONYMY"
"""
'''
NOTE: METONYMY relations mark cross-type metonymies, and will not have
      relation mentions or values for MODALITY and TENSE.  For these
      reasons, we use "relation_mention*" instead of
      "relation_mention+", and "#IMPLIED" for MODALITY and TENSE. 
'''

type_event = """
TYPE="Life" SUBTYPE="Be-Born"
TYPE="Life" SUBTYPE="Die"
TYPE="Life" SUBTYPE="Marry"
TYPE="Life" SUBTYPE="Divorce"
TYPE="Life" SUBTYPE="Injure"
TYPE="Transaction" SUBTYPE="Transfer-Ownership"
TYPE="Transaction" SUBTYPE="Transfer-Money"
TYPE="Movement" SUBTYPE="Transport"
TYPE="Business" SUBTYPE="Start-Org"
TYPE="Business" SUBTYPE="End-Org"
TYPE="Business" SUBTYPE="Declare-Bankruptcy"
TYPE="Business" SUBTYPE="Merge-Org"
TYPE="Conflict" SUBTYPE="Attack"
TYPE="Conflict" SUBTYPE="Demonstrate"
TYPE="Contact" SUBTYPE="Meet"
TYPE="Contact" SUBTYPE="Phone-Write"
TYPE="Personnel" SUBTYPE="Start-Position"
TYPE="Personnel" SUBTYPE="End-Position"
TYPE="Personnel" SUBTYPE="Nominate"
TYPE="Personnel" SUBTYPE="Elect"
TYPE="Justice" SUBTYPE="Arrest-Jail"
TYPE="Justice" SUBTYPE="Release-Parole"
TYPE="Justice" SUBTYPE="Charge-Indict"
TYPE="Justice" SUBTYPE="Trial-Hearing"
TYPE="Justice" SUBTYPE="Sue"
TYPE="Justice" SUBTYPE="Convict"
TYPE="Justice" SUBTYPE="Sentence"
TYPE="Justice" SUBTYPE="Fine"
TYPE="Justice" SUBTYPE="Execute"
TYPE="Justice" SUBTYPE="Extradite"
TYPE="Justice" SUBTYPE="Acquit"
TYPE="Justice" SUBTYPE="Pardon"
TYPE="Justice" SUBTYPE="Appeal"
"""

type_event_role = """
TYPE="Person"
TYPE="Place"
TYPE="Buyer"
TYPE="Seller"
TYPE="Beneficiary"
TYPE="Price"
TYPE="Artifact"
TYPE="Origin"
TYPE="Destination"
TYPE="Giver"
TYPE="Recipient"
TYPE="Money"
TYPE="Org"
TYPE="Agent"
TYPE="Victim"
TYPE="Instrument"
TYPE="Entity"
TYPE="Attacker"
TYPE="Target"
TYPE="Defendant"
TYPE="Adjudicator"
TYPE="Prosecutor"
TYPE="Plaintiff"
TYPE="Crime"
TYPE="Position"
TYPE="Sentence"
TYPE="Vehicle"
TYPE="Time-Within"
TYPE="Time-Starting"
TYPE="Time-Ending"
TYPE="Time-Before"
TYPE="Time-After"
TYPE="Time-Holds"
TYPE="Time-At-Beginning"
TYPE="Time-At-End"
"""

pre_defined_time_role = {'Time-Within',
                         'Time-Starting',
                         'Time-Ending',
                         'Time-Before',
                         'Time-After',
                         'Time-Holds',
                         'Time-At-Beginning',
                         'Time-At-End'}


def get_type_list(text):
    import re
    label_list = list()
    filter_re = re.compile('[\n]+')
    label_text_list = filter_re.split(text)
    for label_text in label_text_list:
        types = label_text.split()
        if len(types) == 1:
            label_list.append(types[0][6:-1])
        elif len(types) == 2:
            label_list.append(types[0][6:-1] + ':' + types[1][9:-1])
        else:
            continue
    return label_list
