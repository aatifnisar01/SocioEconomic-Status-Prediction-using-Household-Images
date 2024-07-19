# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 17:45:10 2022

@author: Aatif
"""

import requests
from tqdm import tqdm

sikkim = ['9. Hatidunga (1051)', '26. Reshi Tenzerbong (1045)', '11. Lower Jaubari (Cheyadara) (1023)', '17. Lower tanak (1063)', '1. Tsozo (1001)', '6. Birkuna Lingyang (1016)', '38. UTokday (1031)', '36. Upper Kamling (1040)', '35. Upper Bokrang (1012)', '28. Singithang (1071)', '25. Pepthang (1050)', '8. Dubdi (Dubdi Sangha Gumpa) (1042)', '4. Sindrabong (1006)', '33. Tinzerbong (1047)', '27. Rimbik (1010)', '22. Nambu (1009)', '13. Lower Kateng (1013)', '10. Jeel (1033)', '7. Deu  (Simkharka) (1039)', '5. Naku (Pemayangtse Gumpa) (1007)', '37. Upper tanak (1064)', '34. Upper Bhaluthang (1028)', '32. Tinik (1036)', '31. Singling gumpadara (1053)', '30. Singling (1055)', '3. Mangthyang (1003)', '29. Singling  ogeng (1057)', '24. Pakjer (1022)', '23. Pabong (1049)', '21. Mazitar compound (1069)', '20. Manpur (1017)', '2. Gangyap (Sinen Ngadak Gum) (1002)', '19. LTokday (1032)', '18. Lower Wok (1073)', '16. Lower Rinchenpong (1030)', '15. Lower Pamphok (1060)', '14. Lower Labing (1015)', '12. Lower Kamling (1041)']


mizoram = ['52. Lalnutui (52)', '99. Puankhai (99)', '98. Thanglailung (98)', '97. Maubawk L (97)', '96. Gulsingbabsora (96)', '95. Phainuam (95)', '94. Zawlpui (94)', '92. Gerasury (92)', '91. Zehtet (91)', '90. Vawmbuk (90)', '9. Gulsil (9)', '89. Tiperaghat I (89)', '88. Songrasury (88)', '86. Bortuli (86)', '85. Lamthai II (85)', '84. Old Tuisumpui (84)', '83. Serhuan (83)', '82. Jaruldulbasora (82)', '81. Lamthai I (81)', '80. Tuichawngtlang (80)', '78. Neihdawn (78)', '77. Sumasumi (77)', '76. Siasi (76)', '75. Tuisen Bolia (75)', '74. Lungtan (74)', '73. Lungsen (73)', '72. Chamdur P I (72)', '71. Bindiasora (71)', '70. Tongkolong (70)', '7. Buknuam (7)', '69. Kalapani (69)', '67. Lokhisuri (67)', '66. Futsury (66)', '65. New.Balukiasuri (65)', '64. Tuipuibari (64)', '63. ThekaduarKawrpuichhuah (63)', '62. Tiperaghat  Ii (62)', '61. Damdep II (New Jognasuri II) (61)', '60. Bymari (60)', '6. Zeropoint (6)', '59. Tawizo (59)', '58. Damdep I (New Jognasuri I) (58)', '57. Andermanik (57)', '56. Damzau I (56)', '55. Balangsuri (55)', '54. Mainababsora II (54)', '53. New Sachan (53)', '51. Mautlang (51)', '50. Diltlang ‘S’ (50)', '5. Silkur (5)', '49. Nunsuri (49)', '48. Laitlang (48)', '47. Lokisuri (47)', '46. Ajasora II (46)', '45. Belthei (45)', '44. Geraguluksora (44)', '43. Diplibagh(Kawizau) (43)', '42. Boronasury (42)', '41. Tiperaghat  III (41)', '40. Borsegojasora (40)', '4. Thinghlun (4)', '39. Chawngte P (39)', '38. Maila (38)', '37. Bornasuri (37)', '36. Ahmypi (36)', '35. Sachan (35)', '34. Sedailui (34)', '33. Pablakhali (33)', '32. Chhotaguisuri I (32)', '31. S.Chawilung (31)', '30. Hmunthar (30)', '3. Muriskata (3)', '29. Chamdurtlang I (29)', '28. Melhnih (Chalrang) (28)', '27. Tuichawngchhuah (27)', '26. Peglababsora (26)', '25. Samuksuri(Chengkawllui) (25)', '24. Kamtuli (24)', '23. Sugarbasora (23)', '22. Gobabsuri (22)', '21. Tuichawng (21)', '20. N. Mualvum (20)', '2. Chhotaguisuri II (2)', '19. Belpei (Matiasora) (19)', '18. Zohmun (18)', '17. Khaikhy (17)', '164. Tongasora (164)', '163. Salmar (163)', '161. Vaitin (161)', '160. S.Sabual (160)', '16. Longmasu (16)', '158. Tlangpui (158)', '157. Hualngohmun (157)', '156. W.Lungdar (156)', '155. Khojoisuri Old (155)', '154. W.Serzawl (154)', '153. Tuirial (153)', '152. Maubawk ‘CH’ (152)', '150. E.Bungtlang (150)', '15. Devasuri  (15)', '148. S.Khawbung (148)', '147. Thenhlum (147)', '146. W.Phulpui (146)', '145. Khojaisurichhuah (145)', '144. Dairy Veng Tuipang (144)', '143. Malsuri (143)', '142. Kaisih (142)', '141. Mar S (141)', '140. Darlawng (140)', '14. Thanzamasora (14)', '139. Thingsen (139)', '138. Serhmun (138)', '136. Rajivnagar1 (136)', '135. Phairuangkai (135)', '134. Chanin (134)', '132. Lower Theiva (132)', '131. Zawlpui (131)', '13. Ajasora (13)', '129. Dengsur (129)', '128. Buhkangkawn (128)', '127. Khojoisuri  (New) (127)', '126. Kawrthah (126)', '125. Falkawn (125)', '124. K Sarali (124)', '123. Lower Sakawrdai (123)', '122. Riangtlei (122)', '121. Sunhluchhip (121)', '12. S.Bungtlang (12)', '119. Mauzam (119)', '118. Charluitlang (118)', '117. Chhumkhum (117)', '116. Boraituli (116)', '115. Putlungasih (115)', '114. W Saizawh (114)', '113. New.Vuakmual  Pangtlang (113)', '112. Samthang (New) (112)', '111. Chengpui (111)', '110. Sailam (110)', '11. Lamthai III (11)', '109. Dapchhuah (Tutphai) (109)', '108. Hliappui S (108)', '107. Bandiasora (107)', '106. Vankal (106)', '105. Kauchhuah (105)', '104. Supha (104)', '103. Sailen (103)', '102. Tlangmawi (102)', '101. Balukiasuri (101)', '100. Bualpui NG W (100)', '10. Belkhai (10)', '1. Vairawkai (1)', '68. Khawlian (68)', '151. Hmuifang (151)', '137. Thaizawl (137)', '133. Sialsuk (133)', '149. South Lungrang (149)', '159. Sihfa (159)', '130. Teirei Forest (130)', '79. Rualalung (79)', '162. N. Kanghmun (162)', '120. Vanhne R (120)', '93. Dilkhan (93)', '87. Tawipui N (87)']

nagaland = ['66. Wui (258)', '32. Chendang  (220)', '30. NEWRISETHSI (218)', '77. Thonoknyu  Vill. (272)', '71. Chilliso (264)', '69. Sangsangnyu Hq (261)', '64. Khudei (256)', '62. Suthazu Nasa (253)', '60. Aniashu (250)', '44. Lofukhong (234)', '38. Sangsomong (228)', '14. Chendang Saddle (196)', '138. Kejok (344)', '133. Meyilong (337)', '126. Lilen (330)', '99. Hanku (301)', '98. Muleangkiur (300)', '97. Kenjenshu (299)', '96. Chare HQ (298)', '95. Khong (297)', '94. Chassir (296)', '93. Changdang (295)', '92. Ngangpong (294)', '91. New Chalkot (293)', '90. Chuhachinglen (291)', '9. Kidima (189)', '89. Yonghong (287)', '88. Lengnyu (286)', '87. Ikiesingram (285)', '86. Yongnyah Village (284)', '85. Moalenden (283)', '84. Chikiponger (281)', '83. Nsenlwa (279)', '82. Chen Hq (277)', '81. K. Longsoru (276)', '80. Old Nkio (275)', '8. Waphur (188)', '79. Sowa Changle (274)', '78. Jakphang (273)', '76. New Puilwa (271)', '75. Longching Village (270)', '74. Chingmei (268)', '73. Mithehe (Mithelijan) (267)', '72. Old Tsadanger (266)', '70. Bamsiakilwa (263)', '7. Nihoi (UR) (187)', '68. Tekuk (UR) (260)', '67. Santsoze (259)', '65. Yimza (257)', '63. Y. Anner Vill (254)', '61. Panso A (252)', '6. Gaili Namdi (186)', '59. Seiyha Phesa (249)', '58. Nyinyem (248)', '57. Nsong (247)', '56. Lochomi (246)', '55. Changlang (245)', '54. Yakhao (244)', '53. Waromong (243)', '52. New Mangakhi (242)', '51. Ikiye (241)', '50. Noksen HQ (240)', '5. Shitovi (185)', '49. Letsam (239)', '48. Wansoi (238)', '47. Khelma (237)', '46. Asukhomi (236)', '45. New Ngaulong (235)', '43. AKUKNEW (233)', '42. Waoshu (232)', '41. Viphoma (231)', '40. Kephore A (230)', '4. Gaili (184)', '39. Khongka (229)', '37. Kejanglwa (227)', '36. Tekivong (226)', "35. Rurur 'A' +'B' (224)", '34. Ngam (223)', '33. Therhutsesemi (221)', '31. Old Puilwa (219)', '3. Ralan (Old) (183)', '29. Sangphur (216)', '28. Holongba (215)', '27. Deukoram (214)', '26. Ngangching (213)', '25. Lasikiur (211)', '24. Jalukie B (210)', '23. Chehozu (UR) (209)', '22. Itovi (207)', '21. Yimrup (206)', '20. Tsurmen (205)', '2. Yangli Mission Centre (182)', '19. Old Chalkot (204)', '18. Ngwalwa (202)', '17. Aree (New) (201)', '16. Old peren (200)', '15. Yakor (197)', '137. Daniel (343)', '136. Chessore Vill. (341)', '135. Angphang (340)', '134. Bongkolong (339)', '132. Chipur (336)', '131. Laokhu (335)', '130. Younyu (334)', '13. Hazadisa (195)', '129. Longshen Hq (333)', '128. New Beisumpui (332)', '127. Shamnyu (331)', '125. Zakho (329)', '124. Nokyan (328)', '123. Tichipami (327)', '122. Tsuwao (326)', '121. Changnyu (325)', '120. Jalukielo (324)', '12. Phaikholum (194)', '119. Thetsumi (323)', '118. Sangchen Compound (UR) (322)', '117. Langnok (320)', '116. Philimi (319)', '115. Nchan (318)', '114. Yakshu (317)', '113. Ndunglwa (316)', '112. Chingtang (315)', '111. Nakshu (314)', '110. Gareiphe Basa (313)', '11. Khulazu Bawe (193)', '109. Noklak Vill. (312)', '108. Langmeang (311)', '107. Jalukiekam (310)', '106. Amosen (UR) (309)', '105. Vongkithem (308)', '104. Nokzang (307)', '103. Nchangram (306)', '102. Zhavame (305)', '101. Sastami (303)', '100. Beisumpuikam (302)', '10. Dungki (192)', '1. Satoi Vill (181)', '140. Heiranglwa (190)', '163. Mpai old N New (282)', '161. Yongphang SComp (UR) (338)', '154. Salomi (265)', '149. Nikihe (212)', '157. Waromong Comp. (251)', '139. Baimho (278)', '153. Sakshi Village (342)', '159. Yehemi (288)', '160. Yimjenkimong (262)', '150. Nzau (289)', '144. Lukuto (217)', '155. Shiwoto (198)', '142. Kuthur (280)', '164. Old Mangakhi (208)', '143. Luhezhe (191)', "141. Khumishi 'A' (292)", '151. Phiro (203)', '145. Mangkolemba Hq (255)', '146. Mungya (225)', '156. Vikheto (199)', '152. Pungren (222)', '162. Maksha (290)', '148. New Tesen (304)', '147. New Land (321)', '158. Yannyu (269)']

tripura = ['12. Ramnagar (521)', '58. Madhabnagar (390)', '99. Karatichhara (443)', '98. Dakshin Laljuri (471)', '97. Dakshin Krishnapur (437)', '96. Dakshin Dhumachhara (446)', '95. Chharthai (453)', '94. Chandipur (441)', '93. Aliamara (455)', '92. Silghati (524)', '91. Paschim Masli (520)', '90. Paschim Magpushkarini (498)', '9. Kanchanmala (382)', '89. Paschim Karamchhara (500)', '88. Kathalchhara (510)', '87. Dakshin Chandrapur (516)', '86. Dakshin Bharatchandranagar (522)', '85. Tuichama (405)', '84. Sonachhara (401)', '83. Ranibari (400)', '82. Paikhola (393)', '81. Murapara (Part) (396)', '80. Madhya Bharatchandranagar (399)', '8. Dasamonipara (372)', '79. Laxmichhara (408)', '78. Baramura Deotamura R.F. (Part) (398)', '77. West Ichailalcherra (429)', '76. Uttar Dhumachhara (425)', '75. Taiharchum (428)', '74. Ranipukur (423)', '73. Manu Chhailengta R.F. (Part) (434)', '72. Chhataria (411)', '71. Chhaigharia (432)', '70. Chapiapara (413)', '7. Kangrai (388)', '13. Dakshin Radhapur (454)', '22. Indurail (366)', '1. Paschim Barjala (394)', '59. Photamati (Part) (367)', '25. Bijoynagar (424)', '23. Noagaon (368)', '45. Dudhpushkarini (502)', '11. Piplacherra (469)', '61. Tekka R.F. (Part) (363)', '69. Uttar Laljuri (474)', '14. Kalapani (463)', '16. Brahmakunda (392)', '31. Dhupcharra (414)', '47. Ishanchandranagar (514)', '101. Mukchhari (460)', '100. Krishnanagar (462)', '105. Ultachhara (470)', '49. Juri R.F (481)', '63. Paschim Manu (483)', '104. Rajdharnagar (439)', '66. Tairbhuma (486)', '35. Paschim Padmabil (519)', '30. Vrigudasbari (501)', '28. Ishanpur (484)', '38. Chandrapur (378)', '102. Purba Potachhara (465)', '48. Jamthumbari (490)', '18. Ramnsankar (404)', '24. Barjala Binapani (389)', '65. Sarala (491)', '6. Champabari (445)', '32. Madhya Champamura (416)', '43. Lalchhara (371)', '33. Uttar Champamura (418)', '42. Kambuk Charra (440)', '34. Paschim Champamura (420)', '36. Uttar Padmabil (523)', '17. Dumrakaridak (395)', '56. Bulongbasa (379)', '29. Barkathal (489)', '44. Bardos (511)', '51. Kalabaria (506)', '5. Damcherra R.F (422)', '4. Zoithang (364)', '10. Mantala (467)', '3. Daksin Padmabil (477)', '67. Tekka R.F.(Part) (478)', '52. South Ganganagar (515)', '64. Paschim Muhuripur (473)', '68. Uttar Ekchhari (476)', '21. Rangacharra (386)', '41. Kalagang (375)', '26. Uttar Deocherra (438)', '60. Silachari (369)', '19. Pekucherra (458)', '37. Bhagyapur (407)', '103. Radhakishorepur R.F. (450)', '20. Chandra Halam Para (461)', '53. Uttar Dasda (499)', '54. Baramura Deotamura R.F. (Part) (391)', '15. Agnipasa (459)', '39. Dhuptali (361)', '2. Dina kobra (435)', '57. Chaimaroa (365)', '62. Ampinagar (479)', '40. Ichailalcharra (427)', '46. Halahali (503)', '106. Uttar Bharatchandranagar (482)', '50. Kakraban (518)', '27. Rabia Sardar (475)', '131. Paschim Tilthai (493)', '118. Kamalpur (417)', '155. Uttar Krishnapur (430)', '154. Ulemchhara (448)', '139. Rahumcherra (485)', '137. Radhapur (456)', '149. Tebaria (383)', '143. Sarat chow. (426)', '120. Kashari R.F. (410)', '147. Singerbil (507)', '124. Mangalkhali (431)', '112. Gangaprasadpara (419)', '122. Lalchhara (436)', '151. Thumsaraipara (480)', '107. Bagpasa (406)', '141. Sabual (447)', '134. Purba Noagoan (517)', '148. Srirampur (494)', '136. Purba Tilthai (412)', '121. Khowaipara (376)', '144. Sardin Khan para (387)', '142. Santipur (402)', '132. Prakashnagar (377)', '135. Purba Peporiakhola (421)', '128. Paschim Anandapur (487)', '153. Twisamongkarui (449)', '127. Palatana (472)', '152. Tuigamari (403)', '156. Vangmun (384)', '138. Raghna (497)', '113. Haripur (385)', '123. Laxminagar (466)', '126. North Ganganagar (513)', '109. Bhairabnagar (464)', '125. Manuchailengta (362)', '116. Jitendra nagar (381)', '111. Ganganagar (Part) (457)', '108. Balidhum (512)', '133. Purba Karamchhara (415)', '140. Rani (508)', '129. Paschim Dewanpasa (488)', '114. Hurijala (492)', '110. Charupasa (504)', '115. Jamjuri (468)', '117. Kacharicherra (370)', '150. Thangsang (444)', '145. Satsangam (496)', '130. Paschim Peporiakhola (495)', '146. Shantinagar (397)', '119. Kanchancherra (380)', '160. Tetaiya (442)', '159. Jubarajnagar (433)', '163. Anangangarar (505)', '162. Karmapara (452)', '164. Gajaria (509)', '158. Joymanipara (409)', '161. Paschim Hmmanpui (451)', '157. Kalagangerpar (374)', '55. Birchandranagar (373)']

import os
os.system('mkdir Mizoram')

for folder in tqdm(mizoram):
    arg = '{"path": '+'"/AMS Data Upload (Photos)/Mizoram/'+folder+'"}'
    arg = arg.encode('utf-8')
    headers = {
        'Authorization': 'DATA NOT PUBLIC ',
        'Dropbox-API-Arg': arg,
    }
    r = requests.post('DATA NOT PUBLIC', headers=headers)
    # print(r)
    save_path = 'Mizoram/'+folder+'.zip'
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
print('Mizoram done.')

os.system('mkdir Nagaland')
#for i in tqdm(range(107, len(nagaland))):
for i in tqdm(range(len(nagaland))):
    folder = nagaland[i]
    arg = '{"path": '+'"/AMS Data Upload (Photos)/Nagaland/'+folder+'"}'
    arg = arg.encode('utf-8')
    headers = {
        'Authorization': 'DATA NOT PUBLIC',
        'Dropbox-API-Arg': arg,
    }
    r = requests.post('DATA NOT PUBLIC', headers=headers)
    save_path = 'Nagaland/'+folder+'.zip'
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
print('Nagaland done.')

os.system('mkdir Tripura')
for folder in tqdm(tripura):
    arg = '{"path": '+'"/AMS Data Upload (Photos)/Tripura/'+folder+'"}'
    arg = arg.encode('utf-8')
    headers = {
        'Authorization': 'DATA NOT PUBLIC',
        'Dropbox-API-Arg': arg,
    }
    r = requests.post('DATA NOT PUBLIC', headers=headers)
    save_path = 'Tripura/'+folder+'.zip'
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
print('Tripura done.')

os.system('mkdir Sikkim')
for folder in tqdm(sikkim):
    arg = '{"path": '+'"/AMS Data Upload (Photos)/Sikkim/'+folder+'"}'
    arg = arg.encode('utf-8')
    headers = {
        'Authorization': 'DATA NOT PUBLIC',
        'Dropbox-API-Arg': arg,
    }
    r = requests.post('DATA NOT PUBLIC', headers=headers)
    save_path = 'Sikkim/'+folder+'.zip'
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
print('Sikkim done.')